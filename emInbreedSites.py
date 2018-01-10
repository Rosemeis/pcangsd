"""
EM algorithm to estimate the per-site inbreeding coefficients for NGS data using genotype likelihoods
and pre-computed allele frequencies (both population or individual).

Simple estimator of the per-site inbreeding coefficients. A likelihood ratio test is also performed for each site.
"""

__author__ = "Jonas Meisner"

# Import help functions
from helpFunctions import *

# Import libraries
import numpy as np
from numba import jit

# Inner update
@jit("void(f4[:, :], f4[:, :], f4[:], f4[:], f4[:])", nopython=True, nogil=True, cache=True)
def innerEM(likeMatrix, indf, expH, expG, F):
	m, n = likeMatrix.shape # Dimension of likelihood matrix
	m /= 3 # Number of individuals
	logTest = np.zeros(n, dtype=np.float32)

	for ind in xrange(m):
		probMatrix = np.empty((3, n), dtype=np.float32)
		for s in xrange(n):
			probMatrix[0, s] = likeMatrix[3*ind, s]*((1 - indf[ind, s])*(1 - indf[ind, s]) + (1 - indf[ind, s])*indf[ind, s]*F[s])
			probMatrix[1, s] = likeMatrix[3*ind + 1, s]*(2*indf[ind, s]*(1 - indf[ind, s])*(1 - F[s]))
			probMatrix[2, s] = likeMatrix[3*ind + 2, s]*(indf[ind, s]*indf[ind, s] + (1 - indf[ind, s])*indf[ind, s]*F[s])
			expH[s] += 2*indf[ind, s]*(1 - indf[ind, s]) # Expected number of heterozygotes
		probMatrix /= np.sum(probMatrix, axis=0)
		expG += probMatrix[1, :] # Sum the posterior of each individual

	for s in xrange(n):
		F[s] = 1 - (expG[s]/expH[s])	

# Loglikelihood estimates
@jit("void(f4[:, :], f4[:, :], f4[:], f4[:], f4[:])", nopython=True, nogil=True, cache=True)
def loglike(likeMatrix, indf, F, logAlt, logNull):
	m, n = likeMatrix.shape # Dimension of likelihood matrix
	m /= 3 # Number of individuals

	for ind in xrange(m):
		likeAlt = np.empty((3, n), dtype=np.float32)
		likeNull = np.empty((3, n), dtype=np.float32)
		for s in xrange(n):
			# Alternative model
			likeAlt[0, s] = likeMatrix[3*ind, s]*((1 - indf[ind, s])*(1 - indf[ind, s]) + (1 - indf[ind, s])*indf[ind, s]*F[s])
			likeAlt[1, s] = likeMatrix[3*ind + 1, s]*(2*indf[ind, s]*(1 - indf[ind, s])*(1 - F[s]))
			likeAlt[2, s] = likeMatrix[3*ind + 2, s]*(indf[ind, s]*indf[ind, s] + (1 - indf[ind, s])*indf[ind, s]*F[s])
			logAlt[s] += np.log(np.sum(likeAlt[:, s]))

			# Null model
			likeNull[0, s] = likeMatrix[3*ind, s]*((1 - indf[ind, s])*(1 - indf[ind, s]))
			likeNull[1, s] = likeMatrix[3*ind + 1, s]*(2*indf[ind, s]*(1 - indf[ind, s]))
			likeNull[2, s] = likeMatrix[3*ind + 2, s]*(indf[ind, s]*indf[ind, s])
			logNull[s] += np.log(np.sum(likeNull[:, s]))

# EM algorithm for estimation of inbreeding coefficients
def inbreedSitesEM(likeMatrix, indf, EM=200, EM_tole=1e-4):
	m, n = likeMatrix.shape # Dimension of likelihood matrix
	m /= 3 # Number of individuals
	F = np.random.rand(n).astype(np.float32) # Random initialization of inbreeding coefficients

	# EM algorithm
	for iteration in xrange(1, EM + 1):
		expG = np.zeros(n, dtype=np.float32) # Container for posterior probability of heterozygosity
		expH = np.zeros(n, dtype=np.float32) # Container for expected heterozygosity
		innerEM(likeMatrix, indf, expH, expG, F) # Update F

		# Break EM update if converged
		if iteration > 1:
			updateDiff = rmse1d(F, F_prev)
			print "Inbreeding coefficients computed	(" +str(iteration) + ") Diff=" + str(updateDiff)
			if updateDiff < EM_tole:
				print "EM (Inbreeding) converged at iteration: " + str(iteration)
				break
		else:
			print "Inbreeding coefficients computed	(" +str(iteration) + ")"

		F_prev = np.copy(F)

	expH = None
	expG = None

	# LRT test statistic
	logAlt = np.zeros(n, dtype=np.float32)
	logNull = np.zeros(n, dtype=np.float32)
	for ind in xrange(m):
		loglike(likeMatrix, indf, F, logAlt, logNull)

	lrt = 2*(logAlt - logNull)

	return F, lrt