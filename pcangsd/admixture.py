"""
PCAngsd.
Estimate admixture proportions and ancestral allele frequencies.
"""

__author__ = "Jonas Meisner"

# libraries
import numpy as np
from math import ceil

# Import scripts
from pcangsd import shared_cy
from pcangsd import admixture_cy

##### Admixture estimation #####
def admixNMF(L, P, K, alpha, iter, tole, batch, seed, verbose, t):
	m, n = P.shape
	np.random.seed(seed)
	shuffleP = np.random.permutation(m)
	X = P[shuffleP,:] # Shuffled individual allele frequencies

	# Initiate factor matrices and containers
	Q = np.random.rand(n, K).astype(np.float32, copy=False)
	Q /= np.sum(Q, axis=1, keepdims=True)
	QA = np.zeros((n, K), dtype=np.float32)
	QB = np.zeros((K, K), dtype=np.float32)
	QC = np.zeros((n, K), dtype=np.float32)
	F = np.random.rand(m, K).astype(np.float32, copy=False)
	FB = np.zeros((K, K), dtype=np.float32)
	frob_vec = np.zeros(m, dtype=np.float32)
	logVec = np.zeros(m, dtype=np.float64)

	# Batch preparation
	m_b = ceil(float(m)/batch)
	bIndex = list(range(0, m, m_b))

	# CSG-MU - cyclic mini-batch stochastic gradient descent
	for i in range(iter):
		Q0 = np.copy(Q)
		for b in bIndex:
			bEnd = min(b + m_b, m)
			Xbatch = X[b:bEnd,:]
			Fbatch = F[b:bEnd,:]
			mInner = Fbatch.shape[0]
			pF = 2*(1 + (n*mInner + n*K)//(mInner*K + mInner))
			pQ = 2*(1 + (n*mInner + mInner*K)//(n*K + n))

			# Update F
			FA = np.dot(Xbatch, Q)
			np.dot(Q.T, Q, out=FB)
			for inner in range(pF):
				F_prev = np.copy(Fbatch)
				FC = np.dot(Fbatch, FB)
				admixture_cy.updateF(Fbatch, FA, FC)
				if inner == 0:
					F_init = shared_cy.frobenius(Fbatch, F_prev)
				else:
					if (shared_cy.frobenius(Fbatch, F_prev) <= (0.1*F_init)):
						break

			# Update Q
			np.dot(Xbatch.T, Fbatch, out=QA)
			np.dot(Fbatch.T, Fbatch, out=QB)
			for inner in range(pQ):
				Q_prev = np.copy(Q)
				np.dot(Q, QB, out=QC)
				admixture_cy.updateQ(Q, QA, QC, alpha)
				Q /= np.sum(Q, axis=1, keepdims=True)
				if inner == 0:
					Q_init = shared_cy.frobenius(Q, Q_prev)
				else:
					if (shared_cy.frobenius(Q, Q_prev) <= 0.1*Q_init):
						break

		# Measure difference
		diff = shared_cy.rmse2d(Q, Q0)
		if verbose:
			print(f"CSG-MU ({i+1}).\tRMSE={np.round(diff,9)}")
		if diff < tole:
			print("Converged.")
			break
	del X, Xbatch, Fbatch, QA, QB, QC, FA, FB, FC, Q_prev, F_prev

	# Reshuffle columns of F
	F = F[np.argsort(shuffleP)]

	# Frobenius and log-likelihood
	X = np.dot(F, Q.T)
	shared_cy.frobeniusThread(X, P, frob_vec, t)
	print(f"Frobenius error: {np.round(np.sqrt(np.sum(frob_vec)),5)}")
	admixture_cy.loglike(L, X, logVec, t)
	logLike = np.sum(logVec, dtype=float)
	print(f"Log-likelihood: {np.round(logLike, 5)}")
	del logVec
	return Q, F, logLike

##### Automatic search for alpha #####
def alphaSearch(L, P, K, aEnd, iter, tole, batch, seed, depth, t):
	# First search
	aMin = 0
	aMax = aEnd
	aMid = (aMin + aMax)/2.0
	aStep = (aMin + aMax)/4.0
	print(f"Depth=1,\tRunning Alpha={aMin}")
	QB, FB, lB = admixNMF(L, P, K, aMin, iter, tole, batch, seed, False, t)
	aL = 0
	aB = aMin
	print(f"Depth=1,\tRunning Alpha={aMid}")
	QT, FT, lT = admixNMF(L, P, K, aMid, iter, tole, batch, seed, False, t)
	if lT > lB:
		QB, FB, lB = np.copy(QT), np.copy(FT), lT
		aL = 1
		aB = aMid
	print(f"Depth=1,\tRunning Alpha={aMax}")
	QT, FT, lT = admixNMF(L, P, K, aMax, iter, tole, batch, seed, False, t)
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
	for d in range(depth-1):
		if aMin == 0:
			print(f"Depth={d+2},\tRunning Alpha={aMid}")
			QT, FT, lT = admixNMF(L, P, K, aMid, iter, tole, batch, seed, False, t)
			if lT > lB:
				QB, FB, lB = np.copy(QT), np.copy(FT), lT
				aL = 1
				aB = aMid
		else:
			print(f"Depth={d+2},\tRunning Alpha={aMin}")
			QT, FT, lT = admixNMF(L, P, K, aMin, iter, tole, batch, seed, False, t)
			if lT > lB:
				QB, FB, lB = np.copy(QT), np.copy(FT), lT
				aL = 0
				aB = aMin
			else:
				print(f"Depth={d+2},\tRunning Alpha={aMax}")
				QT, FT, lT = admixNMF(L, P, K, aMax, iter, tole, batch, seed, False, t)
				if lT > lB:
					QB, FB, lB = np.copy(QT), np.copy(FT), lT
					aL = 2
					aB = aMax
				else:
					argL = 1
		aStep /= 2.0
		if aMin == 0:
			aMax = aMid
			aMid = aMax/2.0
		else:
			aMid = [aMin, aMid, aMax][aL]
			aMin = aMid - aStep
			aMax = aMid + aStep
	return QB, FB, lB, aB
