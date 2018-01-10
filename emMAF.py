"""
EM algorithm to estimate the per-site population allele frequencies for NGS data using genotype likelihoods.
Maximum likelihood estimator.
"""

__author__ = "Jonas Meisner"

# Import libraries
import numpy as np
from numba import jit
import threading
from helpFunctions import *

##### Functions #####
# Calculate posterior genotype probabilities
def updateF(likeMatrix, f, S, N):
	m, n = likeMatrix.shape
	m /= 3
	newMat = np.empty((m, n), dtype=np.float32)

	# Multithreading	
	threads = [threading.Thread(target=innerEM, args=(likeMatrix, f, chunk, N, newMat)) for chunk in S]
	for thread in threads:
		thread.start()
	for thread in threads:
		thread.join()

	newVec = np.sum(newMat, axis=0)/m

	return newVec

# Multithreaded inner update
@jit("void(f4[:, :], f4[:], i8, i8, f4[:, :])", nopython=True, nogil=True, cache=True)
def innerEM(likeMatrix, f, S, N, newMat):
	m, n = likeMatrix.shape # Dimension of likelihood matrix
	m /= 3 # Number of individuals

	for ind in xrange(S, min(S+N, m)):
		probMatrix = np.empty((3, n), dtype=np.float32)
		for s in xrange(n):
			probMatrix[0, s] = likeMatrix[3*ind, s]*((1 - f[s])*(1 - f[s]))
			probMatrix[1, s] = likeMatrix[3*ind + 1, s]*(2*f[s]*(1 - f[s]))
			probMatrix[2, s] = likeMatrix[3*ind + 2, s]*(f[s]*f[s])
		probMatrix /= np.sum(probMatrix, axis=0)
		newMat[ind] = probMatrix[1, :]/2 + probMatrix[2, :]

# EM algorithm for estimation of population allele frequencies
def alleleEM(likeMatrix, EM=200, EM_tole=1e-4, threads=1):
	m, n = likeMatrix.shape
	m /= 3
	f = np.random.rand(n).astype(np.float32) # Uniform initialization
	f.clip(min=1.0/(2*m), max=1-(1.0/(2*m)), out=f)

	# Prepare for multithreading
	chunk_N = int(np.ceil(float(m)/threads))
	chunks = [i * chunk_N for i in xrange(threads)]

	for iteration in xrange(1, EM + 1): # EM iterations
		f = updateF(likeMatrix, f, chunks, chunk_N) # Updated allele frequencies

		# Break EM update if converged
		if iteration > 1:
			diff = rmse1d(f, f_prev)
			if diff < EM_tole:
				print "EM (MAF) converged at iteration: " + str(iteration)
				break

		if iteration == 2:
			oldDiff = diff
		elif iteration > 2:
			# Second convergence criterion
			if abs(diff - oldDiff) <= 1e-5:
				print "Estimation of individual allele frequencies has converged " + str(iteration) + ". RMSD between iterations: " + str(abs(diff - oldDiff))
				break
			else:
				oldDiff = diff

		f_prev = np.copy(f)
	return f