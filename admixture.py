"""
Estimate admixture using Non-negative Matrix Factorization based on multiplicative updates.
"""

__author__ = "Jonas Meisner"

# Import libraries
import numpy as np
import admixture_cy
import shared

# Estimate admixture using non-negative matrix factorization
def admixNMF(L, Pi, K, alpha, admix_iter, admix_tole, seed, batch, verbose, t):
	n, m = Pi.shape

	# Shuffle individual allele frequencies
	np.random.seed(seed) # Set random seed
	shuffleX = np.random.permutation(m)
	X = Pi[:, shuffleX]

	# Initiate factor matrices and containers
	Q = np.random.rand(n, K).astype(np.float32, copy=False)
	Q /= np.sum(Q, axis=1, keepdims=True)
	prevQ = np.copy(Q)
	F = np.random.rand(m, K).astype(np.float32, copy=False)
	loglike_vec = np.zeros(m, dtype=np.float32)

	# Batch preparation
	batch_M = int(np.ceil(float(m)/batch))
	bIndex = np.arange(0, m, batch_M)

	# CSG-MU - Cyclic mini-batch stochastic gradient descent
	for iteration in range(1, admix_iter+1):
		for b in bIndex:
			bEnd = min(b + batch_M, m)
			Xbatch = X[:, b:bEnd]
			Fbatch = F[b:bEnd]
			mInner = Fbatch.shape[0]
			pF = 2*(1 + (n*mInner + n*K)//(mInner*K + mInner))
			pQ = 2*(1 + (n*mInner + mInner*K)//(n*K + n))

			# Update F
			A = np.dot(Xbatch.T, Q)
			B = np.dot(Q.T, Q)
			for inner in range(pF): # Acceleration updates
				F_prev = np.copy(Fbatch)
				FB = np.dot(Fbatch, B)
				admixture_cy.updateF(Fbatch, A, FB, t)
				if inner == 0:
					F_init = shared.frobenius(Fbatch, F_prev)
				else:
					if (shared.frobenius(Fbatch, F_prev) <= (0.1*F_init)):
						break

			# Update Q
			A = np.dot(Xbatch, Fbatch)
			B = np.dot(Fbatch.T, Fbatch)
			for inner in range(pQ): # Acceleration updates
				Q_prev = np.copy(Q)
				QB = np.dot(Q, B)
				admixture_cy.updateQ(Q, A, QB, alpha, t)
				Q /= np.sum(Q, axis=1, keepdims=True)

				if inner == 0:
					Q_init = shared.frobenius(Q, Q_prev)
				else:
					if (shared.frobenius(Q, Q_prev) <= (0.1*Q_init)):
						break

		# Measure difference
		diff = shared.rmse2d(Q, prevQ)
		if verbose:
			print("CSG-MU (" + str(iteration) + "). Q-RMSD=" + str(diff))

		if diff < admix_tole:
			print("CSG-MU has converged.")
			break
		prevQ = np.copy(Q)

	del X, Xbatch, prevQ, A, B, FB, QB, F_prev, Q_prev

	# Reshuffle columns
	F = F[np.argsort(shuffleX)]

	# Frobenius and log-like
	X = np.dot(Q, F.T)
	admixture_cy.clipX(X, t)
	cost = shared.frobenius(X, Pi)
	print("Frobenius error: " + str(cost))

	admixture_cy.loglike(L, X, loglike_vec, t)
	logLike = np.sum(loglike_vec)
	print("Log-likelihood: " + str(logLike))
	return Q, F, logLike

# Automatic search for appropriate alpha
def alphaSearch(L, Pi, K, aEnd, admix_iter, admix_tole, seed, batch, depth, t):
	# First search
	aMin = 0
	aMax = aEnd
	aMid = (aMin + aMax)/2.0
	aStep = (aMin + aMax)/4.0

	print("NMF: K=" + str(K) + ", alpha=" + str(aMin))
	Q_best, F_best, L_best = admixNMF(L, Pi, K, aMin, admix_iter, admix_tole, seed, batch, False, t)
	argL = 0
	aBest = aMin
	print("NMF: K=" + str(K) + ", alpha=" + str(aMid))
	Q_test, F_test, L_test = admixNMF(L, Pi, K, aMid, admix_iter, admix_tole, seed, batch, False, t)
	if L_test > L_best:
		Q_best, F_best, L_best = np.copy(Q_test), np.copy(F_test), L_test
		argL = 1
		aBest = aMid
	print("NMF: K=" + str(K) + ", alpha=" + str(aMax))
	Q_test, F_test, L_test = admixNMF(L, Pi, K, aMax, admix_iter, admix_tole, seed, batch, False, t)
	if L_test > L_best:
		Q_best, F_best, L_best = np.copy(Q_test), np.copy(F_test), L_test
		argL = 2
		aBest = aMax
	
	if argL == 0:
		aMax = aMid
		aMid = aMid/2.0
	else:
		aMid = [aMin, aMid, aMax][argL]
		aMin = aMid - aStep
		aMax = aMid + aStep

	for d in range(2, depth+1):
		print("\nDepth=" + str(d) + ", best alpha=" + str(aBest) + ", log-like=" + str(L_best))
		if aMin == 0:
			print("NMF: K=" + str(K) + ", alpha=" + str(aMid))
			Q_test, F_test, L_test = admixNMF(L, Pi, K, aMid, admix_iter, admix_tole, seed, batch, False, t)
			if L_test > L_best:
				Q_best, F_best, L_best = np.copy(Q_test), np.copy(F_test), L_test
				argL = 1
				aBest = aMid
		else:
			print("NMF: K=" + str(K) + ", alpha=" + str(aMin))
			Q_test, F_test, L_test = admixNMF(L, Pi, K, aMin, admix_iter, admix_tole, seed, batch, False, t)
			if L_test > L_best:
				Q_best, F_best, L_best = np.copy(Q_test), np.copy(F_test), L_test
				argL = 0
				aBest = aMin

			else:
				print("NMF: K=" + str(K) + ", alpha=" + str(aMax))
				Q_test, F_test, L_test = admixNMF(L, Pi, K, aMax, admix_iter, admix_tole, seed, batch, False, t)
				if L_test > L_best:
					Q_best, F_best, L_best = np.copy(Q_test), np.copy(F_test), L_test
					argL = 2
					aBest = aMax
				else:
					argL = 1
		aStep /= 2.0
		if aMin == 0:
			aMax = aMid
			aMid = aMax/2.0
		else:
			aMid = [aMin, aMid, aMax][argL]
			aMin = aMid - aStep
			aMax = aMid + aStep
	return Q_best, F_best, L_best, aBest