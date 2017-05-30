"""
Call genotypes from using estimated individual allele frequencies as prior.
"""

__author__ = "Jonas Meisner"

# Import libraries
import numpy as np

# Genotype caller
def callGeno(likeMatrix, f):
	mTotal, n = likeMatrix.shape # Dimension of likelihood matrix
	m = mTotal/3 # Number of individuals
	G = np.zeros((m, n), dtype=int)

	# Estimate posteior probabilities
	if f.ndim == 1:
		# Genotype frequencies based on individual allele frequencies under HWE
		fMatrix = np.vstack(((1-f)**2, 2*f*(1-f), f**2))

	for ind in range(m):
		if f.ndim == 2:
			# Genotype frequencies based on individual allele frequencies under HWE 
			fMatrix = np.vstack(((1-f[ind])**2, 2*f[ind]*(1-f[ind]), f[ind]**2))
			
		wLike = likeMatrix[(3*ind):(3*ind+3)]*fMatrix # Weighted likelihoods
		gProp = wLike/np.sum(wLike, axis=0) # Genotype probabilities of individual
		nanSites = np.isnan(np.sum(gProp, axis=0))

		# Find genotypes with highest probability
		genos = np.argmax(gProp, axis=0)
		genos[nanSites] = 3
		G[ind] = genos 

	return G