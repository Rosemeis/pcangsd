"""
EM algorithm to estimate individual inbreeding coefficients for NGS data using genotype likelihoods
and pre-computed allele frequencies (both sample average or individual).

Both the maximum likelihood estimator and simple estimator of the inbreeding coefficients can be computed.
The estimators can be selected by the model parameter (model=1 for MLE, model=2 for Simple).
"""

__author__ = "Jonas Meisner"

# Import help functions
from helpFunctions import *

# Import libraries
import numpy as np


# EM algorithm for estimation of inbreeding coefficients
def inbreedEM(likeMatrix, f, model=1, EM=200, EM_tole=1e-4):
	mTotal, n = likeMatrix.shape # Dimension of likelihood matrix
	m = mTotal/3 # Number of individuals
	F = np.random.rand(m) # Random intialization of inbreeding coefficients
	F_prev = np.ones(m)*np.inf # Initiate likelihood measure to infinity

	if model == 1: # Maximum likelihood estimator
		for iteration in range(EM): # EM iterations
			if f.ndim == 1:
				# Estimated genotype frequencies given IBD (Z)
				fMatrix_z0 = np.vstack(((1-f)**2, 2*f*(1-f), f**2))
				fMatrix_z1 = np.vstack(((1-f), np.zeros(n), f))

			wLike = np.zeros((2,n)) # Weighted likelihood by prior

			for ind in range(m):
				if f.ndim == 2:
					# Estimated genotype frequencies given IBD (Z)
					fMatrix_z0 = np.vstack(((1-f[ind])**2, 2*f[ind]*(1-f[ind]), f[ind]**2))
					fMatrix_z1 = np.vstack(((1-f[ind]), np.zeros(n), f[ind]))

				wLike[0, :] = np.sum(likeMatrix[(3*ind):(3*ind+3)]*fMatrix_z0, axis=0)*(1-F[ind])
				wLike[1, :] = np.sum(likeMatrix[(3*ind):(3*ind+3)]*fMatrix_z1, axis=0)*F[ind]
			
				# Expectation maximation - Update F
				zProb = wLike/np.sum(wLike, axis=0) 
				F[ind] = np.sum(zProb[1, :])/float(n)

			# Break EM update if converged
			if rmse(F, F_prev) < EM_tole:
				print "EM (Inbreeding) converged at iteration: " + str(iteration)
				break

			F_prev = np.copy(F)


	elif model == 2: # Secondary model - Simple estimator (Vieira-model)
		for iteration in range(EM): # EM iterations
			if f.ndim == 1:
				# Expected number of heterozygotes
				expH = np.sum(2*f*(1-f))

			for ind in range(m):
				if f.ndim == 1:
					# Estimated genotype frequencies given F
					fMatrix = np.vstack(((1-f)**2 + (1-f)*f*F[ind], 2*(1-f)*f*(1-F[ind]), f**2 + (1-f)*f*F[ind]))
				
				else:
					# Estimated genotype frequencies given F
					fMatrix = np.vstack(((1-f[ind])**2 + (1-f[ind])*f[ind]*F[ind], 2*(1-f[ind])*f[ind]*(1-F[ind]), f[ind]**2 + (1-f[ind])*f[ind]*F[ind]))
					
					# Expected number of heterozygotes
					expH = np.sum(2*f[ind]*(1-f[ind]))

				wLike = likeMatrix[(3*ind):(3*ind+3)]*fMatrix # Weighted likelihood by prior

				# Expectation maximation - Update F
				gProb = wLike/np.sum(wLike, axis=0)
				F[ind] = 1 - np.sum(gProb[1, :])/expH

			# Break EM update if converged
			if rmse(F, F_prev) < EM_tole:
				print "EM (Inbreeding) converged at iteration: " + str(iteration)
				break

			F_prev = np.copy(F)

	return F