"""
EM algorithm to estimate the per-site inbreeding coefficients for NGS data using genotype likelihoods
and pre-computed allele frequencies (both sample average or individual).

Both the maximum likelihood estimator and simple estimator of the inbreeding coefficients can be computed.
The estimators can be selected by the model parameter (model=1 for MLE, model=2 for Simple).
"""

__author__ = "Jonas Meisner"

# Import help functions
from helpFunctions import *

# Import libraries
import numpy as np


# EM algorithm for estimation of minor allele frequencies
def inbreedSitesEM(likeMatrix, f, model=1, EM=1000, EM_tole=1e-6):
	mTotal, n = likeMatrix.shape # Dimension of likelihood matrix
	m = mTotal/3 # Number of individuals
	F = np.random.rand(n) # Random intialization of inbreeding coefficients
	F_prev = np.ones(n)*np.inf # Initiate likelihood measure to infinity

	if model == 1: # Maximum likelihood estimator
		for iteration in range(EM): # EM iterations
			if f.ndim == 1:
				# Estimated genotype frequencies given IBD (Z)
				fMatrix_z0 = np.vstack(((1-f)**2, 2*f*(1-f), f**2))
				fMatrix_z1 = np.vstack(((1-f), np.zeros(n), f))
		
			wLike = np.zeros((2,n)) # Weighted likelihood by prior
			expZ = np.zeros(n) # Posterior expectation of IBD

			for ind in range(m):
				if f.ndim > 1:
					# Estimated genotype frequencies given IBD (Z)
					fMatrix_z0 = np.vstack(((1-f[ind])**2, 2*f[ind]*(1-f[ind]), f[ind]**2))
					fMatrix_z1 = np.vstack(((1-f[ind]), np.zeros(n), f[ind]))

				wLike[0,:] = np.sum(likeMatrix[(3*ind):(3*ind+3)]*fMatrix_z0, axis=0)*(1-F) # Z = 0
				wLike[1,:] = np.sum(likeMatrix[(3*ind):(3*ind+3)]*fMatrix_z1, axis=0)*F # Z = 1
				zProb = wLike/np.sum(wLike, axis=0) # Z probabilities

				# E-step
				expZ += zProb[1,:] # Sum the posterior of each individual
		
			F = expZ/float(m) # Updated inbreeding coefficients (M-step)

			# Break EM update if converged
			if rmse(F, F_prev) < EM_tole:
				print "EM (Inbreeding) converged at iteration: " + str(iteration)
				break

			F_prev = F


	elif model == 2: # Secondary model - Simple estimator (Vieira-model)
		for iteration in range(EM): # EM iterations
			if f.ndim == 1:
				# Estimated genotype frequencies given F
				fMatrix = np.vstack(((1-f)**2 + (1-f)*f*F, 2*f*(1-f)*(1-F), f**2 + (1-f)*f*F))

				# Expected number of heterozygotes
				expH = float(m)*2*f*(1-f)
			else:
				expH = np.zeros(n)
		
			expG1 = np.zeros(n) # Posterior of heterozygosity

			for ind in range(m):
				if f.ndim > 1:
					# Estimated genotype frequencies given F
					fMatrix = np.vstack(((1-f[ind])**2 + (1-f[ind])*f[ind]*F, 2*f[ind]*(1-f[ind])*(1-F), f[ind]**2 + (1-f[ind])*f[ind]*F))

					# Expected number of heterozygotes
					expH += 2*f[ind]*(1-f[ind])

				wLike = likeMatrix[(3*ind):(3*ind+3)]*fMatrix # Weighted likelihood by prior
				gProb = wLike/np.sum(wLike, axis=0) # Genotype probabilities

				# E-step
				expG1 += gProb[1,:] # Sum the posterior of each individual
		
			F = 1 - (expG1/expH) # Updated inbreeding coefficients (M-step)

			# Break EM update if converged
			if rmse(F, F_prev) < EM_tole:
				print "EM (Inbreeding) converged at iteration: " + str(iteration)
				break

			F_prev = F

	return F