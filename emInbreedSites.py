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
from math import log

# Inner update
@jit("void(f4[:, :], f4[:, :], f8[:])", nopython=True, nogil=True, cache=True)
def innerEM(likeMatrix, indf, F):
	m, n = likeMatrix.shape # Dimension of likelihood matrix
	m /= 3 # Number of individuals
	probMatrix = np.zeros((3, n))
	expG = np.zeros(n) # Container for posterior probability of heterozygosity
	expH = np.zeros(n) # Container for expected heterozygosity

	for ind in xrange(m):
		# Estimate posterior probabilities
		for s in xrange(n):
			probMatrix[0, s] = likeMatrix[3*ind, s]*((1 - indf[ind, s])*(1 - indf[ind, s]) + (1 - indf[ind, s])*indf[ind, s]*F[s])
			probMatrix[1, s] = likeMatrix[3*ind + 1, s]*2*indf[ind, s]*(1 - indf[ind, s])*(1 - F[s])
			probMatrix[2, s] = likeMatrix[3*ind + 2, s]*(indf[ind, s]*indf[ind, s] + (1 - indf[ind, s])*indf[ind, s]*F[s])
			expH[s] += 2*indf[ind, s]*(1 - indf[ind, s]) # Expected number of heterozygotes

			for g in xrange(3):
				probMatrix[g, s] = max(0.0001, probMatrix[g, s])
		probMatrix /= np.sum(probMatrix, axis=0)
		expG += probMatrix[1, :] # Sum the posterior of each individual

	for s in xrange(n):
		F[s] = 1 - (expG[s]/expH[s])

# Loglikelihood estimates
@jit("void(f4[:, :], f4[:, :], f8[:], f8[:], f8[:])", nopython=True, nogil=True, cache=True)
def loglike(likeMatrix, indf, F, logAlt, logNull):
	m, n = likeMatrix.shape # Dimension of likelihood matrix
	m /= 3 # Number of individuals
	likeAlt = np.zeros((3, n))
	likeNull = np.zeros((3, n))

	for ind in xrange(m):
		for s in xrange(n):
			# Alternative model
			likeAlt[0, s] = likeMatrix[3*ind, s]*((1 - indf[ind, s])*(1 - indf[ind, s]) + (1 - indf[ind, s])*indf[ind, s]*F[s])
			likeAlt[1, s] = likeMatrix[3*ind + 1, s]*(2*indf[ind, s]*(1 - indf[ind, s])*(1 - F[s]))
			likeAlt[2, s] = likeMatrix[3*ind + 2, s]*(indf[ind, s]*indf[ind, s] + (1 - indf[ind, s])*indf[ind, s]*F[s])

			for g in xrange(3):
				likeAlt[g, s] = max(0.0001, likeAlt[g, s])
			logAlt[s] += np.log(np.sum(likeAlt[:, s]))

			# Null model
			likeNull[0, s] = likeMatrix[3*ind, s]*(1 - indf[ind, s])*(1 - indf[ind, s])
			likeNull[1, s] = likeMatrix[3*ind + 1, s]*2*indf[ind, s]*(1 - indf[ind, s])
			likeNull[2, s] = likeMatrix[3*ind + 2, s]*indf[ind, s]*indf[ind, s]
			logNull[s] += np.log(np.sum(likeNull[:, s]))

# EM algorithm for estimation of inbreeding coefficients
def inbreedSitesEM(likeMatrix, indf, EM=200, EM_tole=1e-4):
	m, n = likeMatrix.shape # Dimension of likelihood matrix
	m /= 3 # Number of individuals
	F = np.ones(n)*0.25 # Initialization of inbreeding coefficients
	F_prev = np.copy(F)

	# EM algorithm
	for iteration in xrange(1, EM + 1):
		innerEM(likeMatrix, indf, F) # Update F

		# Break EM update if converged
		updateDiff = rmse1d(F, F_prev)
		print "Inbreeding coefficients estimated (" +str(iteration) + "). RMSD=" + str(updateDiff)
		if updateDiff < EM_tole:
			print "EM (Inbreeding - sites) converged at iteration: " + str(iteration)
			break

		F_prev = np.copy(F)

	# LRT test statistic
	logAlt = np.zeros(n)
	logNull = np.zeros(n)
	loglike(likeMatrix, indf, F, logAlt, logNull)

	lrt = 2*(logAlt - logNull)

	return F, lrt