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
	probVec = np.zeros(3) # Container for posterior probabilities
	priorVec = np.empty(3) # Container for prior probabilities

	for s in xrange(S, min(S+N, n)):
		expH = 0
		tempVec = np.empty(3)

		for ind in xrange(m):
			Fadj = (1 - indF[ind, s])*indF[ind, s]*F[s]
			priorVec[0] = max(1e-4, (1 - indF[ind, s])*(1 - indF[ind, s]) + Fadj)
			priorVec[1] = max(1e-4, 2*indF[ind, s]*(1 - indF[ind, s]) - 2*Fadj)
			priorVec[2] = max(1e-4, indF[ind, s]*indF[ind, s] + Fadj)
			priorVec /= np.sum(priorVec)

			# Estimate posterior probabilities
			tempVec[0] = likeMatrix[3*ind, s]*priorVec[0]
			tempVec[1] = likeMatrix[3*ind + 1, s]*priorVec[1]
			tempVec[2] = likeMatrix[3*ind + 2, s]*priorVec[2]
			tempVec /= np.sum(tempVec)

			# Counts of heterozygotes
			expH += 2*indF[ind, s]*(1 - indF[ind, s])
			probVec += tempVec

		# ANGSD procedure
		probVec /= m
		probVec[0] = max(1e-4, probVec[0])
		probVec[1] = max(1e-4, probVec[1])
		probVec[2] = max(1e-4, probVec[2])
		probVec /= np.sum(probVec)

		# Update the inbreeding coefficient
		F[s] = 1 - (m*probVec[1]/expH)
		F[s] = max(-1.0, F[s])
		F[s] = min(1.0, F[s])

# Loglikelihood estimates
@jit("void(f4[:, :], f4[:, :], f8[:], i8, i8, f8[:], f8[:])", nopython=True, nogil=True, cache=True)
def loglike(likeMatrix, indF, F, S, N, logAlt, logNull):
	m, n = indF.shape # Dimensions
	likeAlt = np.zeros(3)
	likeNull = np.zeros(3)
	priorVec = np.empty(3) # Container for prior probabilities

	for s in xrange(S, min(S+N, n)):
		for ind in xrange(m):
			### Alternative model
			# Priors
			Fadj = (1 - indF[ind, s])*indF[ind, s]*F[s]
			priorVec[0] = max(1e-4, (1 - indF[ind, s])*(1 - indF[ind, s]) + Fadj)
			priorVec[1] = max(1e-4, 2*indF[ind, s]*(1 - indF[ind, s]) - 2*Fadj)
			priorVec[2] = max(1e-4, indF[ind, s]*indF[ind, s] + Fadj)
			priorVec /= np.sum(priorVec)

			# Posteriors
			likeAlt[0] = likeMatrix[3*ind, s]*priorVec[0]
			likeAlt[1] = likeMatrix[3*ind + 1, s]*priorVec[1]
			likeAlt[2] = likeMatrix[3*ind + 2, s]*priorVec[2]
			logAlt[s] += log(np.sum(likeAlt))

			### Null model
			likeNull[0] = likeMatrix[3*ind, s]*(1 - indF[ind, s])*(1 - indF[ind, s])
			likeNull[1] = likeMatrix[3*ind + 1, s]*2*indF[ind, s]*(1 - indF[ind, s])
			likeNull[2] = likeMatrix[3*ind + 2, s]*indF[ind, s]*indF[ind, s]
			logNull[s] += log(np.sum(likeNull))

# EM algorithm for estimation of inbreeding coefficients
def inbreedSitesEM(likeMatrix, indF, EM=200, EM_tole=1e-4, t=1):
	m, n = indF.shape # Dimensions
	F = np.zeros(n) # Initialization of inbreeding coefficients
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

	lrt = 2*(logAlt - logNull)

	return F, lrt