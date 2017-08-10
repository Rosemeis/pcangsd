"""
Call genotypes from using estimated individual allele frequencies as prior.
"""

__author__ = "Jonas Meisner"

# Import libraries
import numpy as np

# Genotype caller
def callGeno(likeMatrix, f, delta, F=None):
	mTotal, n = likeMatrix.shape # Dimension of likelihood matrix
	m = mTotal/3 # Number of individuals
	G = np.zeros((m, n))

	# Estimate posteior probabilities
	for ind in range(m):
		if type(F) != type(None):
			# Genotype frequencies based on individual allele frequencies and inbreeding (HWE extended)
			fMatrix = np.vstack(((1-f[ind])**2 + f[ind]*(1-f[ind])*F[ind], 2*f[ind]*(1-f[ind])*(1-F[ind]), f[ind]**2 + f[ind]*(1-f[ind])*F[ind]))			
		else:
			# Genotype frequencies based on individual allele frequencies under HWE 
			fMatrix = np.vstack(((1-f[ind])**2, 2*f[ind]*(1-f[ind]), f[ind]**2))
		wLike = likeMatrix[(3*ind):(3*ind+3)]*fMatrix # Weighted likelihoods
		gProp = wLike/np.sum(wLike, axis=0) # Genotype probabilities of individual

		# Find genotypes with highest probability
		genos = np.argmax(gProp, axis=0)
		nanSites = gProp[genos, np.arange(n)] < delta
		G[ind] = genos
		G[ind, nanSites] = np.nan

	return G