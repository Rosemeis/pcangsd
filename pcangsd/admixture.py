import numpy as np
from math import ceil
from pcangsd import shared_cy
from pcangsd import admixture_cy

##### Admixture estimation #####
def admixNMF(L, P, K, alpha, iter, tole, batch, rng, verbose):
	M, N = P.shape
	M_b = ceil(float(M)/batch)
	S_i = rng.permuted(np.arange(M, dtype=np.uint32))
	X = P[S_i,:] # Shuffled individual allele frequencies

	# Initiate factor matrices and containers
	F = rng.random(size=(M, K), dtype=np.float32)
	Q = rng.random(size=(N, K), dtype=np.float32)
	Q /= np.sum(Q, axis=1, keepdims=True)
	Q0 = np.zeros_like(Q)
	QA = np.zeros_like(Q)
	QC = np.zeros_like(Q)
	FB = np.zeros((K, K), dtype=np.float32)
	QB = np.zeros((K, K), dtype=np.float32)

	# Batch preparation

	# CSG-MU - cyclic mini-batch stochastic gradient descent
	for i in np.arange(iter):
		memoryview(Q0.ravel())[:] = memoryview(Q.ravel())
		for b in np.arange(batch):
			bEnd = min((b+1)*M_b, M)
			X_batch = X[(b*M_b):bEnd,:]
			F_batch = F[(b*M_b):bEnd,:]
			M_i = F_batch.shape[0]
			pF = 2*(1 + (N*M_i + N*K)//(M_i*K + M_i))
			pQ = 2*(1 + (N*M_i + M_i*K)//(N*K + N))

			# Update F
			FA = np.dot(X_batch, Q)
			np.dot(Q.T, Q, out=FB)
			for inner in np.arange(pF):
				F_prev = np.copy(F_batch)
				FC = np.dot(F_batch, FB)
				admixture_cy.updateF(F_batch, FA, FC)
				if inner == 0:
					F_init = shared_cy.frobenius(F_batch, F_prev)
				else:
					if (shared_cy.frobenius(F_batch, F_prev) <= (0.1*F_init)):
						break

			# Update Q
			np.dot(X_batch.T, F_batch, out=QA)
			np.dot(F_batch.T, F_batch, out=QB)
			for inner in np.arange(pQ):
				Q_prev = np.copy(Q)
				np.dot(Q, QB, out=QC)
				admixture_cy.updateQ(Q, QA, QC, alpha)
				if inner == 0:
					Q_init = shared_cy.frobenius(Q, Q_prev)
				else:
					if (shared_cy.frobenius(Q, Q_prev) <= 0.1*Q_init):
						break

		# Measure difference
		diff = shared_cy.rmse2d(Q, Q0)
		if verbose:
			print(f"CSG-MU ({i+1}).\tRMSE={diff:.9f}")
		if diff < tole:
			print("Converged.")
			break
	del X, X_batch, F_batch, QA, QB, QC, FA, FB, FC, Q_prev, F_prev

	# Reshuffle columns of F
	F = F[np.argsort(S_i)]

	# Frobenius and log-likelihood
	X = np.dot(F, Q.T)
	F_err = shared_cy.frobeniusMulti(X, P)
	L_lik = admixture_cy.loglike(L, X)
	print(f"Frobenius error: {F_err:.5f}")
	print(f"Log-likelihood: {L_lik:.5f}")
	return Q, F, L_lik

##### Automatic search for alpha #####
def alphaSearch(L, P, K, aEnd, iter, tole, batch, depth, rng):
	# First search
	aMin = 0
	aMax = aEnd
	aMid = (aMin + aMax)/2.0
	aStep = (aMin + aMax)/4.0
	print(f"Depth=1,\tRunning Alpha={aMin}")
	QB, FB, lB = admixNMF(L, P, K, aMin, iter, tole, batch, rng, False)
	aL = 0
	aB = aMin
	print(f"Depth=1,\tRunning Alpha={aMid}")
	QT, FT, lT = admixNMF(L, P, K, aMid, iter, tole, batch, rng, False)
	if lT > lB:
		QB, FB, lB = np.copy(QT), np.copy(FT), lT
		aL = 1
		aB = aMid
	print(f"Depth=1,\tRunning Alpha={aMax}")
	QT, FT, lT = admixNMF(L, P, K, aMax, iter, tole, batch, rng, False)
	if lT > lB:
		QB, FB, lB = np.copy(QT), np.copy(FT), lT
		aL = 2
		aB = aMax

	# Prepare new step
	if aL == 0:
		aMax = aMid
		aMid = aMid/2.0
	else:
		aMid = [aMin, aMid, aMax][aL]
		aMin = aMid - aStep
		aMax = aMid + aStep
	for d in np.arange(depth-1):
		if aMin == 0:
			print(f"Depth={d+2},\tRunning Alpha={aMid}")
			QT, FT, lT = admixNMF(L, P, K, aMid, iter, tole, batch, rng, False)
			if lT > lB:
				memoryview(QB.ravel())[:] = memoryview(QT)
				memoryview(FB.ravel())[:] = memoryview(FT)
				lB = lT
				aL = 1
				aB = aMid
		else:
			print(f"Depth={d+2},\tRunning Alpha={aMin}")
			QT, FT, lT = admixNMF(L, P, K, aMin, iter, tole, batch, rng, False)
			if lT > lB:
				memoryview(QB.ravel())[:] = memoryview(QT)
				memoryview(FB.ravel())[:] = memoryview(FT)
				lB = lT
				aL = 0
				aB = aMin
			else:
				print(f"Depth={d+2},\tRunning Alpha={aMax}")
				QT, FT, lT = admixNMF(L, P, K, aMax, iter, tole, batch, rng, False)
				if lT > lB:
					memoryview(QB.ravel())[:] = memoryview(QT)
					memoryview(FB.ravel())[:] = memoryview(FT)
					lB = lT
					aL = 2
					aB = aMax
				else:
					aL = 1
		aStep /= 2.0
		if aMin == 0:
			aMax = aMid
			aMid = aMax/2.0
		else:
			aMid = [aMin, aMid, aMax][aL]
			aMin = aMid - aStep
			aMax = aMid + aStep
	return QB, FB, lB, aB
