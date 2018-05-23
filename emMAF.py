"""
EM algorithm to estimate the per-site population allele frequencies for NGS data using genotype likelihoods.
Maximum likelihood estimator.
"""

__author__ = "Jonas Meisner"

# Import libraries
import numpy as np
from numba import jit
import threading
from helpFunctions import rmse1d

##### Functions #####
# Calculate posterior genotype probabilities
def updateF(likeMatrix, f, S, N):
	m, n = likeMatrix.shape
	m /= 3
	newF = np.zeros(n)

	# Multithreading
	threads = [threading.Thread(target=innerEM, args=(likeMatrix, f, chunk, N, newF)) for chunk in S]
	for thread in threads:
		thread.start()
	for thread in threads:
		thread.join()

	return newF

# Multithreaded inner update
@jit("void(f4[:, :], f8[:], i8, i8, f8[:])", nopython=True, nogil=True, cache=True)
def innerEM(likeMatrix, f, S, N, newF):
	m, n = likeMatrix.shape # Dimension of likelihood matrix
	m /= 3 # Number of individuals

	for s in xrange(S, min(S+N, n)):
		for ind in xrange(m):
			p0 = likeMatrix[3*ind, s]*(1 - f[s])*(1 - f[s])
			p1 = likeMatrix[3*ind + 1, s]*2*f[s]*(1 - f[s])
			p2 = likeMatrix[3*ind + 2, s]*f[s]*f[s]
			newF[s] += (p1 + 2*p2)/(2*(p0 + p1 + p2))
		newF[s] /= m

# EM algorithm for estimation of population allele frequencies
def alleleEM(likeMatrix, EM=200, EM_tole=1e-4, threads=1):
	m, n = likeMatrix.shape
	m /= 3
	f = np.ones(n)*0.25 # Uniform initialization

	# Prepare for multithreading
	chunk_N = int(np.ceil(float(n)/threads))
	chunks = [i * chunk_N for i in xrange(threads)]

	for iteration in xrange(1, EM + 1): # EM iterations
		f = updateF(likeMatrix, f, chunks, chunk_N) # Updated allele frequencies

		# Break EM update if converged
		if iteration > 1:
			diff = rmse1d(f, f_prev)
			if diff < EM_tole:
				print "EM (MAF) converged at iteration: " + str(iteration)
				break
		f_prev = np.copy(f)
	return f
