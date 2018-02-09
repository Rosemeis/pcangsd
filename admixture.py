"""
Estimate admixture using Non-negative Matrix Factorization based on multiplicative updates.
"""

__author__ = "Jonas Meisner"

# Import libraries
import numpy as np
from numba import jit
from helpFunctions import *
from math import log

##### Functions #####
# Estimate log likelihood of ngsAdmix model (inner)
@jit("void(f4[:, :], f8[:, :], i8, i8, f8[:])", nopython=True, nogil=True, cache=True)
def logLike_admixInner(likeMatrix, X, S, N, L):
	m, n = likeMatrix.shape
	m /= 3
	for ind in xrange(S, min(S+N, m)):
		temp = np.empty(3)
		for s in xrange(n):
			temp[0] = likeMatrix[3*ind, s]*(1 - X[ind, s])*(1 - X[ind, s])
			temp[1] = likeMatrix[3*ind+1, s]*2*X[ind, s]*(1 - X[ind, s])
			temp[2] = likeMatrix[3*ind+2, s]*X[ind, s]*X[ind, s]
			L[ind] += log(np.sum(temp))

# Estimate log likelihood of ngsAdmix model (outer)
def logLike_admix(likeMatrix, X, chunks, chunk_N):
	m, n = likeMatrix.shape
	m /= 3
	logLike_inds = np.zeros(m)

	# Multithreading
	threads = [threading.Thread(target=logLike_admixInner, args=(likeMatrix, X, chunk, chunk_N, logLike_inds)) for chunk in chunks]
	for thread in threads:
		thread.start()
	for thread in threads:
		thread.join()

	return np.sum(logLike_inds)


# Estimate admixture using non-negative matrix factorization
def admixNMF(X, K, likeMatrix, alpha=0, iter=50, tole=1e-5, seed=0, batch=20, threads=1):
	m, n = X.shape # Dimensions of individual allele frequencies

	# Initiate matrices
	np.random.seed(seed) # Set random seed
	Q = np.random.rand(m, K)
	Q.clip(min=1e-7, max=1-(1e-7), out=Q)
	Q /= np.sum(Q, axis=1, keepdims=True)
	F = np.random.rand(n, K)
	F.clip(min=1e-7, max=1-(1e-7), out=F)
	
	# Shuffle individual allele frequencies
	shuffleX = np.random.permutation(n)
	X = X[:, shuffleX]
	Xnorm = frobeniusSingle(X)

	# Multithreading
	chunk_N = int(np.ceil(float(m)/threads))
	chunks = [i * chunk_N for i in xrange(threads)]

	# Batch preparation
	batchSize = batch
	batch_N = int(np.ceil(float(n)/batchSize))
	bIndex = np.arange(0, n, batch_N)
	alphaBatch = alpha

	# ASG-MU
	for iteration in xrange(1, iter + 1):
		perm = np.random.permutation(batchSize)

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
				F.clip(min=1e-7, max=1-(1e-7), out=F)

				if inner == 0:
					F_init = frobenius(F[b:bEnd], F_prev)
				else:
					if (frobenius(F[b:bEnd], F_prev) <= (0.1*F_init)):
						break
			del F_prev
			
			# Update Q
			A = np.dot(Xbatch, F[b:bEnd])
			B = np.dot(F[b:bEnd].T, F[b:bEnd])
			for inner in xrange(pQ): # Acceleration updates
				Q_prev = np.copy(Q)
				Q *= A/(np.dot(Q, B) + alphaBatch)
				Q /= np.sum(Q, axis=1, keepdims=True)
				Q.clip(min=1e-7, max=1-(1e-7), out=Q)

				if inner == 0:
					Q_init = frobenius(Q, Q_prev)
				else:
					if (frobenius(Q, Q_prev) <= (0.1*Q_init)):
						break
			del Q_prev

		# Measure difference
		if iteration > 1:
			relObj = frobenius2d_multi(X, np.dot(Q, F.T), chunks, chunk_N)/Xnorm
			print "ASG-MU (" + str(iteration) + "). Relative objective: " + str(relObj)
		
			if abs(relObj - oldObj) < tole:
				print "Admixture estimation has converged. Relative difference between iterations: " + str(abs(relObj - oldObj))
				break
			elif (relObj > oldObj) & (batchSize != 1):
				batchFactor = batchSize/float(batchSize/2)
				batchSize /= 2
				batch_N = int(np.ceil(float(n)/batchSize))
				bIndex = np.arange(0, n, batch_N)
				alphaBatch *= batchFactor
				print "B=" + str(batchSize) + ", alpha=" + str(alphaBatch) + " due to positive difference between iterations"
			oldObj = relObj

		else:
			oldObj = frobenius2d_multi(X, np.dot(Q, F.T), chunks, chunk_N)/Xnorm
			print "ASG-MU (" + str(iteration) + ")"

	del perm
	del A
	del B

	F = F[np.argsort(shuffleX)]
	logLike = logLike_admix(likeMatrix, np.dot(Q, F.T), chunks, chunk_N)
	print "Log-likelihood: " + str(logLike)
	return Q, F, logLike