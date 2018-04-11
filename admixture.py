"""
Estimate admixture using Non-negative Matrix Factorization based on multiplicative updates.
"""

__author__ = "Jonas Meisner"

# Import libraries
import numpy as np
from numba import jit
from helpFunctions import *
from math import log
from scipy.sparse.linalg import svds

##### Functions #####
# Estimate log likelihood of ngsAdmix model (inner)
@jit("void(f4[:, :], f8[:, :], i8, i8, f8[:])", nopython=True, nogil=True, cache=True)
def logLike_admixInner(likeMatrix, X, S, N, L):
	m, n = likeMatrix.shape
	m /= 3
	temp = np.empty(3) # Container for each genotype
	for ind in xrange(S, min(S+N, m)):
		for s in xrange(n):
			temp[0] = likeMatrix[3*ind, s]*(1 - X[ind, s])*(1 - X[ind, s])
			temp[1] = likeMatrix[3*ind+1, s]*2*X[ind, s]*(1 - X[ind, s])
			temp[2] = likeMatrix[3*ind+2, s]*X[ind, s]*X[ind, s]
			L[ind] += log(np.sum(temp))

# Estimate log likelihood of ngsAdmix model (outer)
def logLike_admix(likeMatrix, X, chunks, chunk_N):
	m, n = likeMatrix.shape
	m /= 3
	logLike_inds = np.zeros(m) # Log-likelihood container for each individual

	# Multithreading
	threads = [threading.Thread(target=logLike_admixInner, args=(likeMatrix, X, chunk, chunk_N, logLike_inds)) for chunk in chunks]
	for thread in threads:
		thread.start()
	for thread in threads:
		thread.join()

	return np.sum(logLike_inds)

# Update factor matrices
@jit("void(f8[:, :], f8[:, :], f8[:, :])", nopython=True, nogil=True, cache=True)
def updateF(F, A, FB):
	n, K = F.shape
	for s in xrange(n):
		for k in xrange(K):
			F[s, k] *= A[s, k]/FB[s, k]
			F[s, k] = max(F[s, k], 1e-4)
			F[s, k] = min(F[s, k], 1-(1e-4))

@jit("void(f8[:, :], f8[:, :], f8[:, :], f8)", nopython=True, nogil=True, cache=True)
def updateQ(Q, A, QB, alpha):
	m, K = Q.shape
	for i in xrange(m):
		for k in xrange(K):
			Q[i, k] *= A[i, k]/(QB[i, k] + alpha)
			Q[i, k] = max(Q[i, k], 1e-4)
			Q[i, k] = min(Q[i, k], 1-(1e-4))


# Estimate admixture using non-negative matrix factorization
def admixNMF(X, K, likeMatrix, alpha=0, iter=100, tole=5e-5, seed=0, batch=5, threads=1):
	m, n = X.shape # Dimensions of individual allele frequencies

	# Shuffle individual allele frequencies
	np.random.seed(seed) # Set random seed
	shuffleX = np.random.permutation(n)
	X = X[:, shuffleX]

	# Initiate matrices
	Q = np.zeros((m, K))
	randInit = np.random.permutation(n)[:(K*20)]
	for init in xrange(20):
		for k in xrange(K):
			Q[:, k] += X[:, randInit[K*init + k]]

	Q /= np.sum(Q, axis=1, keepdims=True)
	F = np.dot(np.linalg.inv(np.dot(Q.T, Q)), np.dot(Q.T, X)).T
	F.clip(min=1e-4, max=1-(1e-4), out=F)
	prevQ = np.copy(Q)

	# Multithreading
	chunk_N = int(np.ceil(float(m)/threads))
	chunks = [i * chunk_N for i in xrange(threads)]

	# Batch preparation
	batch_N = int(np.ceil(float(n)/batch))
	bIndex = np.arange(0, n, batch_N)

	# ASG-MU
	for iteration in xrange(1, iter + 1):
		perm = np.random.permutation(batch)

		for b in bIndex[perm]:
			bEnd = min(b + batch_N, n)
			Xbatch = X[:, b:bEnd]
			nInner = Xbatch.shape[1]
			pQ = 2*(1 + (m*nInner + nInner*K)/(m*K + m))

			# Update F
			F[b:bEnd] = np.dot(np.linalg.inv(np.dot(Q.T, Q)), np.dot(Q.T, Xbatch)).T
			F[b:bEnd].clip(min=1e-4, max=1-(1e-4), out=F[b:bEnd])
			
			# Update Q
			A = np.dot(Xbatch, F[b:bEnd])
			B = np.dot(F[b:bEnd].T, F[b:bEnd])
			for inner in xrange(pQ): # Acceleration updates
				Q_prev = np.copy(Q)
				updateQ(Q, A, np.dot(Q, B), alpha)
				Q /= np.sum(Q, axis=1, keepdims=True)

				if inner == 0:
					Q_init = frobenius(Q, Q_prev)
				else:
					if (frobenius(Q, Q_prev) <= (0.1*Q_init)):
						break
			del Q_prev

		# Measure difference
		diff = rmse2d_multi(Q, prevQ, chunks, chunk_N)
		print "ASG-MU (" + str(iteration) + "). Q-RMSD=" + str(diff)
		
		if diff < tole:
			print "Admixture estimation has converged."
			break
		prevQ = np.copy(Q)

	del A, B, perm
	
	# Reshuffle
	F = F[np.argsort(shuffleX)]
	X = X[:, np.argsort(shuffleX)]
	
	# Frobenius and log-like
	Xhat = np.dot(Q, F.T)
	Obj = frobenius2d_multi(X, Xhat, chunks, chunk_N)
	print "Frobenius error: " + str(Obj)

	logLike = logLike_admix(likeMatrix, Xhat, chunks, chunk_N) # Log-likelihood (ngsAdmix model)
	print "Log-likelihood: " + str(logLike)
	return Q, F