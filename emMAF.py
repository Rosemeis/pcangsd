"""
EM algorithm to estimate the per-site sample allele frequencies for NGS data using genotype likelihoods.
Maximum likelihood estimator.
"""

__author__ = "Jonas Meisner"

# Import help functions
from helpFunctions import *

# Import libraries
import numpy as np


# EM algorithm for estimation of minor allele frequencies
def alleleEM(likeMatrix, EM=1000, EM_tole=1e-6):
	mTotal, n = likeMatrix.shape
	m = mTotal/3
	f = np.random.rand(n) # Uniform initialization
	f_prev = np.ones(n)*np.inf # Initiate likelihood measure to infinity

	for iteration in range(EM): # EM iterations
		fMatrix = np.vstack(((1-f)**2, 2*f*(1-f), f**2)) # Estimated genotype frequencies under HWE
		expG = np.zeros(n) # Posterior expectation of genotype

		for ind in range(m):
			wLike = likeMatrix[(3*ind):(3*ind+3)]*fMatrix # Weighted likelihood by prior
			gProb = wLike/np.sum(wLike, axis=0) # Genotype probabilities

			# E-step
			expG += gProb[1,:]/2 + gProb[2,:] # Sum the posterior of each individual
		
		f = expG/float(m) # Updated allele frequencies (M-step)

		# Break EM update if converged
		if rmse(f, f_prev) < EM_tole:
			print "EM (MAF) converged at iteration: " + str(iteration)
			break

		f_prev = np.copy(f)

	return f