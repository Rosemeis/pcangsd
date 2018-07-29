"""
EM algorithm to estimate the per-site inbreeding coefficients for NGS data using genotype likelihoods
and pre-computed allele frequencies (both population or individual).

Simple estimator of the per-site inbreeding coefficients. A likelihood ratio test is also performed for each site.
"""

__author__ = "Jonas Meisner"

# Import help functions
from helpFunctions import rmse1d

# Import libraries
import numpy as np
import threading
from numba import jit
from math import log

# Inner update
@jit("void(f4[:, :], f4[:, :], i8, i8, f8[:])", nopython=True, nogil=True, cache=True)
def inbreedSitesEM_inner(likeMatrix, indF, S, N, F):
	m, n = indF.shape # Dimensions
	probMatrix = np.empty((3, m)) # Container for posterior probabilities

	for s in xrange(S, min(S+N, n)):
		expH = 0

		# Estimate posterior probabilities
		for ind in xrange(m):
			probMatrix[0, ind] = max(0, likeMatrix[3*ind, s]*((1 - indF[ind, s])*(1 - indF[ind, s]) + (1 - indF[ind, s])*indF[ind, s]*F[s]))
			probMatrix[1, ind] = max(0, likeMatrix[3*ind + 1, s]*2*indF[ind, s]*(1 - indF[ind, s])*(1 - F[s]))
			probMatrix[2, ind] = max(0, likeMatrix[3*ind + 2, s]*(indF[ind, s]*indF[ind, s] + (1 - indF[ind, s])*indF[ind, s]*F[s]))
			sumNorm = np.sum(probMatrix[:, ind])

			# Normalize posteriors
			if sumNorm > 0:
				probMatrix[1, ind] /= sumNorm
			probMatrix[1, ind] = max(1e-4, probMatrix[1, ind]) # Fix lower boundary

			# Expected number of heterozygotes
			expH += 2*indF[ind, s]*(1 - indF[ind, s])
		
		# Update the inbreeding coefficient
		F[s] = 1 - (np.sum(probMatrix[1, :])/expH)

# Loglikelihood estimates
@jit("void(f4[:, :], f4[:, :], f8[:], i8, i8, f8[:], f8[:])", nopython=True, nogil=True, cache=True)
def loglike(likeMatrix, indF, F, S, N, logAlt, logNull):
	m, n = indF.shape # Dimensions
	likeAlt = np.zeros(3)
	likeNull = np.zeros(3)

	for s in xrange(S, min(S+N, n)):
		for ind in xrange(m):
			# Alternative model
			likeAlt[0] = max(0, likeMatrix[3*ind, s]*((1 - indF[ind, s])*(1 - indF[ind, s]) + (1 - indF[ind, s])*indF[ind, s]*F[s]))
			likeAlt[1] = max(0, likeMatrix[3*ind + 1, s]*(2*indF[ind, s]*(1 - indF[ind, s])*(1 - F[s])))
			likeAlt[2] = max(0, likeMatrix[3*ind + 2, s]*(indF[ind, s]*indF[ind, s] + (1 - indF[ind, s])*indF[ind, s]*F[s]))
			sumAlt = np.sum(likeAlt)

			if sumAlt > 0:
				logAlt[s] += log(sumAlt)

			# Null model
			likeNull[0] = likeMatrix[3*ind, s]*(1 - indF[ind, s])*(1 - indF[ind, s])
			likeNull[1] = likeMatrix[3*ind + 1, s]*2*indF[ind, s]*(1 - indF[ind, s])
			likeNull[2] = likeMatrix[3*ind + 2, s]*indF[ind, s]*indF[ind, s]
			logNull[s] += log(np.sum(likeNull))

# EM algorithm for estimation of inbreeding coefficients
def inbreedSitesEM(likeMatrix, indF, EM=200, EM_tole=1e-4, t=1):
	m, n = indF.shape # Dimensions
	F = np.ones(n)*0.25 # Initialization of inbreeding coefficients
	F_prev = np.copy(F)

	# Multithreading parameters
	chunk_N = int(np.ceil(float(n)/t))
	chunks = [i * chunk_N for i in xrange(t)]

	# EM algorithm
	for iteration in xrange(1, EM + 1):
		# Multithreading - Update F
		threads = [threading.Thread(target=inbreedSitesEM_inner, args=(likeMatrix, indF, chunk, chunk_N, F)) for chunk in chunks]
		for thread in threads:
			thread.start()
		for thread in threads:
			thread.join()

		# Break EM update if converged
		updateDiff = rmse1d(F, F_prev)
		print "Inbreeding coefficients estimated (" + str(iteration) + "). RMSD=" + str(updateDiff)
		if updateDiff < EM_tole:
			print "EM (Inbreeding - sites) converged at iteration: " + str(iteration)
			break
		F_prev = np.copy(F)

	# LRT test statistic
	logAlt = np.zeros(n)
	logNull = np.zeros(n)

	# Multithreading - Estimate log-likelihoods of two models
	threads = [threading.Thread(target=loglike, args=(likeMatrix, indF, F, chunk, chunk_N, logAlt, logNull)) for chunk in chunks]
	for thread in threads:
		thread.start()
	for thread in threads:
		thread.join()

	lrt = 2*logAlt - 2*logNull

	return F, lrt