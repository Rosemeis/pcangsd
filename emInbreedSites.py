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


# EM algorithm for estimation of inbreeding coefficients
def inbreedSitesEM(likeMatrix, f, EM=200, EM_tole=1e-4):
	mTotal, n = likeMatrix.shape # Dimension of likelihood matrix
	m = mTotal/3 # Number of individuals
	F = np.random.rand(n) # Random initialization of inbreeding coefficients
	F_prev = np.ones(n)*np.inf # Initiate likelihood measure to infinity
	logAlt = 0
	logNull = 0

	for iteration in range(EM): # EM iterations
		if f.ndim == 1:
			# Estimated genotype frequencies given F
			fMatrix = np.vstack(((1-f)**2 + (1-f)*f*F, 2*f*(1-f)*(1-F), f**2 + (1-f)*f*F))
			f0 = np.vstack(((1-f)**2, 2*f*(1-f), f**2)) # F=0

			# Expected number of heterozygotes
			expH = float(m)*2*f*(1-f)
		else:
			expH = np.zeros(n)
		
		expG1 = np.zeros(n) # Container for posterior of heterozygosity

		for ind in range(m):
			if f.ndim == 2:
				# Estimated genotype frequencies given F
				fMatrix = np.vstack(((1-f[ind])**2 + (1-f[ind])*f[ind]*F, 2*f[ind]*(1-f[ind])*(1-F), f[ind]**2 + (1-f[ind])*f[ind]*F))

				# Expected number of heterozygotes
				expH += 2*f[ind]*(1-f[ind])

			wLike = likeMatrix[(3*ind):(3*ind+3)]*fMatrix # Weighted likelihood by prior
			gProb = wLike/np.sum(wLike, axis=0) # Posterior probabilities

			# E-step
			expG1 += gProb[1, :] # Sum the posterior of each individual
		
		F = 1 - (expG1/expH) # Updated inbreeding coefficients (M-step)

		# Break EM update if converged
		updateDiff = rmse(F, F_prev)
		print "Inbreeding coefficients computed	(" +str(iteration) + ") Diff=" + str(updateDiff)
		if updateDiff < EM_tole:
			print "EM (Inbreeding) converged at iteration: " + str(iteration)
			break
		elif iteration == (EM-1):
			print "EM (Inbreeding) was stopped at " + str(iteration) + " with diff: " + str(updateDiff)

		F_prev = np.copy(F)

	# LRT test statistic
	for ind in range(m):
		if f.ndim == 2:
			# Estimated genotype frequencies given F
			fMatrix = np.vstack(((1-f[ind])**2 + (1-f[ind])*f[ind]*F, 2*f[ind]*(1-f[ind])*(1-F), f[ind]**2 + (1-f[ind])*f[ind]*F))
			f0 = np.vstack(((1-f[ind])**2, 2*f[ind]*(1-f[ind]), f[ind]**2)) # F=0

		wLike = likeMatrix[(3*ind):(3*ind+3)]*fMatrix # Weighted likelihood by prior
		w0 = likeMatrix[(3*ind):(3*ind+3)]*f0 # Weighted likelihood of null model
		logAlt += np.log(np.sum(wLike, axis=0)) # Log-likelihood for individual
		logNull += np.log(np.sum(w0, axis=0)) # Log-likelihood for individual in null model

	lrt = 2*(logAlt - logNull)

	return F, lrt