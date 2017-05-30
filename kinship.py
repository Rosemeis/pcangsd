"""
Kinship estimator using genotype likelihoods based on Conomos et al. (2016).
"""

__author__ = "Jonas Meisner"

# Import libraries
import numpy as np

# Kinship estimator
def kinshipConomos(likeMatrix, f):
	mTotal, n = likeMatrix.shape # Dimension of likelihood matrix
	m = mTotal/3 # Number of individuals
	num = np.zeros((m, n)) # Container for numerator in estimation
	numDiag = np.zeros(m) # Container for diagonal of the numerator
	dem = np.zeros((m, n)) # Container for denominator in estimation
	gVector = np.array([0,1,2]) # Genotype vector

	for ind in range(m):
		# Genotype frequencies based on individual allele frequencies under HWE 
		fMatrix = np.vstack(((1-f[ind])**2, 2*f[ind]*(1-f[ind]), f[ind]**2))
			
		wLike = likeMatrix[(3*ind):(3*ind+3)]*fMatrix # Weighted likelihoods
		gProp = wLike/np.sum(wLike, axis=0) # Genotype probabilities of individual
		gProp = np.nan_to_num(gProp) # Set NaNs to 0

		# Setting up for matrix multiplication
		num[ind] = np.sum((((gVector*np.ones((n, 3))).T - 2*f[ind])*gProp), axis=0)
		dem[ind] = np.sqrt(f[ind]*(1-f[ind]))

		numTemp = (((gVector*np.ones((n, 3))).T - 2*f[ind])*gProp)
		numDiag[ind] = np.trace(np.dot(numTemp, numTemp.T))

	phi = np.dot(num, num.T)
	np.fill_diagonal(phi, numDiag)
	phi = phi/(4*np.dot(dem, dem.T))

	return phi