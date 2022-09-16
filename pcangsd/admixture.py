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
	Q_A = np.zeros((n, K), dtype=np.float32)
	Q_B = np.zeros((K, K), dtype=np.float32)
	Q_C = np.zeros((n, K), dtype=np.float32)
	F = np.random.rand(m, K).astype(np.float32, copy=False)
	F_B = np.zeros((K, K), dtype=np.float32)
	frob_vec = np.zeros(m, dtype=np.float32)
	logl_vec = np.zeros(m, dtype=np.float32)

	# Batch preparation
	batch_M = ceil(float(m)/batch)
	bIndex = list(range(0, m, batch_M))

	# CSG-MU - cyclic mini-batch stochastic gradient descent
	for i in range(iter):
		prevQ = np.copy(Q)
		for b in bIndex:
			bEnd = min(b + batch_M, m)
			Xbatch = X[b:bEnd,:]
			Fbatch = F[b:bEnd,:]
			mInner = Fbatch.shape[0]
			pF = 2*(1 + (n*mInner + n*K)//(mInner*K + mInner))
			pQ = 2*(1 + (n*mInner + mInner*K)//(n*K + n))

			# Update F
			F_A = np.dot(Xbatch, Q)
			np.dot(Q.T, Q, out=F_B)
			for inner in range(pF):
				F_prev = np.copy(Fbatch)
				F_C = np.dot(Fbatch, F_B)
				admixture_cy.updateF(Fbatch, F_A, F_C)
				if inner == 0:
					F_init = shared_cy.frobenius(Fbatch, F_prev)
				else:
					if (shared_cy.frobenius(Fbatch, F_prev) <= (0.1*F_init)):
						break

			# Update Q
			np.dot(Xbatch.T, Fbatch, out=Q_A)
			np.dot(Fbatch.T, Fbatch, out=Q_B)
			for inner in range(pQ):
				Q_prev = np.copy(Q)
				np.dot(Q, Q_B, out=Q_C)
				admixture_cy.updateQ(Q, Q_A, Q_C, alpha)
				Q /= np.sum(Q, axis=1, keepdims=True)
				if inner == 0:
					Q_init = shared_cy.frobenius(Q, Q_prev)
				else:
					if (shared_cy.frobenius(Q, Q_prev) <= 0.1*Q_init):
						break

		# Measure difference
		diff = shared_cy.rmse2d(Q, prevQ)
		if verbose:
			print("CSG-MU (" + str(i+1) + "). RMSE=" + str(diff))
		if diff < tole:
			print("Converged.")
			break
	del X, Xbatch, Fbatch, Q_A, Q_B, Q_C, F_A, F_B, F_C, Q_prev, F_prev

	# Reshuffle columns of F
	F = F[np.argsort(shuffleP)]

	# Frobenius and log-likelihood
	X = np.dot(F, Q.T)
	shared_cy.frobeniusThread(X, P, frob_vec, t)
	print("Frobenius error: " + str(np.sqrt(np.sum(frob_vec))))
	admixture_cy.loglike(L, X, logl_vec, t)
	loglike = np.sum(logl_vec, dtype=float)
	print("Log-likelihood: " + str(loglike))
	del logl_vec
	return Q, F, loglike

##### Automatic search for alpha #####
def alphaSearch(L, P, K, aEnd, iter, tole, batch, seed, depth, t):
	# First search
	aMin = 0
	aMax = aEnd
	aMid = (aMin + aMax)/2.0
	aStep = (aMin + aMax)/4.0
	print("Depth=1, Running Alpha=" + str(aMin))
	QB, FB, lB = admixNMF(L, P, K, aMin, iter, tole, batch, seed, False, t)
	aL = 0
	aB = aMin
	print("Depth=1, Running Alpha=" + str(aMid))
	QT, FT, lT = admixNMF(L, P, K, aMid, iter, tole, batch, seed, False, t)
	if lT > lB:
		QB, FB, lB = np.copy(QT), np.copy(FT), lT
		aL = 1
		aB = aMid
	print("Depth=1, Running Alpha=" + str(aMax))
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
			print("Depth=" + str(d+2) + ", Running Alpha=" + str(aMid))
			QT, FT, lT = admixNMF(L, P, K, aMid, iter, tole, batch, seed, False, t)
			if lT > lB:
				QB, FB, lB = np.copy(QT), np.copy(FT), lT
				aL = 1
				aB = aMid
		else:
			print("Depth=" + str(d+2) + ", Running Alpha=" + str(aMin))
			QT, FT, lT = admixNMF(L, P, K, aMin, iter, tole, batch, seed, False, t)
			if lT > lB:
				QB, FB, lB = np.copy(QT), np.copy(FT), lT
				aL = 0
				aB = aMin
			else:
				print("Depth=" + str(d+2) + ", Running Alpha=" + str(aMax))
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
