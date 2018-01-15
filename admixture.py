"""
Estimate admixture using Non-negative Matrix Factorization based on multiplicative updates.
"""

__author__ = "Jonas Meisner"

# Import libraries
import numpy as np
from helpFunctions import *

##### Functions #####
# Estimate admixture using non-negative matrix factorization
def admixNMF(X, K, cov, alpha=0, iter=50, tole=1e-5, seed=0, batch=20, threads=1):
	np.random.seed(seed) # Set random seed
	m, n = X.shape # Dimension of likelihood matrix
	Q = np.random.rand(m, K).astype(np.float32)
	Q /= np.sum(Q, axis=1, keepdims=True)
	F = np.random.rand(n, K).astype(np.float32)
	Q.clip(min=1e-7, max=1-(1e-7), out=Q)
	F.clip(min=1e-7, max=1-(1e-7), out=F)
	shuffleX = np.random.permutation(n)
	X = X[:, shuffleX]
	Xnorm = frobeniusSingle(X)

	# Multithreading
	chunk_N = int(np.ceil(float(m)/threads))
	chunks = [i * chunk_N for i in xrange(threads)]

	# Batch preparation
	batch_N = int(np.ceil(float(n)/batch))
	bIndex = np.arange(0, n, batch_N)

	# NMF - ASG-MU
	for iteration in xrange(1, iter + 1):
		perm = np.random.permutation(batch)

		for b in bIndex[perm]:
			bEnd = min(b + batch_N, n)
			Xbatch = X[:, b:bEnd]
			nInner = Xbatch.shape[1]
			pF = 2*(1 + (m*nInner + m*K)/(nInner*K + nInner))
			pQ = 2*(1 + (m*nInner + nInner*K)/(m*K + m))
			
			# Update F
			A = np.dot(Xbatch.T, Q)
			B = np.dot(Q.T, Q)
			for inner in xrange(pF): # Acceleration updates
				F_prev = np.copy(F[b:bEnd])
				F[b:bEnd] *= A/np.dot(F[b:bEnd], B)
				F[b:bEnd].clip(min=1e-7, max=1-(1e-7), out=F[b:bEnd])

				if inner == 0:
					F_init = frobenius(F[b:bEnd], F_prev)
				else:
					if (frobenius(F[b:bEnd], F_prev) <= (0.1*F_init)):
						break

			# Update Q
			A = np.dot(Xbatch, F[b:bEnd])
			B = np.dot(F[b:bEnd].T, F[b:bEnd])
			for inner in xrange(pQ): # Acceleration updates
				Q_prev = np.copy(Q)
				Q *= A/(np.dot(Q, B) + alpha)
				Q.clip(min=1e-7, out=Q)
				Q /= np.sum(Q, axis=1, keepdims=True)

				if inner == 0:
					Q_init = frobenius(Q, Q_prev)
				else:
					if (frobenius(Q, Q_prev) <= (0.1*Q_init)):
						break

		# Measure difference
		if iteration > 1:
			relObj = frobenius2d_multi(X, np.dot(Q, F.T), chunks, chunk_N)/Xnorm
			print "Admixture has been computed (" + str(iteration) + "). Relative objective: " + str(relObj)
		
			if abs(relObj - oldObj) < tole:
				print "NMF (Admixture) has converged. Difference between iterations: " + str(abs(relObj - oldObj))
				break
			oldObj = relObj

		else:
			oldObj = frobenius2d_multi(X, np.dot(Q, F.T), chunks, chunk_N)/Xnorm
			print "Admixture has been computed (" + str(iteration) + ")"

	X = X[:, np.argsort(shuffleX)]
	F = F[np.argsort(shuffleX)]
	return Q, F