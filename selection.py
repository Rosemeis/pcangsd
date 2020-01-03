"""
Selection scan using principal components based on Galinsky et al. (2016)

Outputs the chi-square distributed selection statistics for each of the top
principal components.
"""

__author__ = "Jonas Meisner"

# Import libraries
import numpy as np
import covariance_cy
import shared
from scipy.sparse.linalg import svds

# Selection scan
def selectionScan(L, Pi, f, K, t):
	n, m = Pi.shape # Dimensions
	E = np.empty((n, m), dtype=np.float32)
	Dsquared = np.empty((m, K), dtype=np.float32)
	covariance_cy.updatePCAngsd(L, Pi, E, t)
	covariance_cy.standardizeE(E, f, t)

	# Performing SVD on normalized expected genotypes
	_, _, U = svds(E, k=K)
	U = U[::-1, :]
	shared.computeD(U, Dsquared)
	return Dsquared


# SNP weights (un-normalized selection scan)
def snpWeights(L, Pi, f, K, t):
	n, m = Pi.shape # Dimensions
	E = np.empty((n, m), dtype=np.float32)
	covariance_cy.updatePCAngsd(L, Pi, E, t)
	covariance_cy.standardizeE(E, f, t)

	# Performing SVD on normalized expected genotypes
	_, s, U = svds(E, k=K)
	snpW = U[::-1, :].T*(s[::-1]**2)/m # Scaling by eigenvalues (PC-scores)
	return snpW