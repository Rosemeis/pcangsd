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

# Selection scan - Varimax rotated basis
def varimaxScan(L, Pi, f, K, t):
	n, m = Pi.shape # Dimensions
	E = np.empty((n, m), dtype=np.float32)
	Dsquared = np.empty((m, K), dtype=np.float32)
	covariance_cy.updatePCAngsd(L, Pi, E, t)
	covariance_cy.standardizeE(E, f, t)	

	# Performing SVD on normalized expected genotypes
	_, _, U = svds(E, k=K)
	U = U[::-1, :]
	Ut, R = varimaxRotation(U)
	shared.computeD(Ut.T, Dsquared)
	return Dsquared, R

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

# Varimax rotation (https://en.wikipedia.org/wiki/Talk:Varimax_rotation)
def varimaxRotation(P):
	k, m = P.shape
	R = np.eye(k, dtype=np.float32)
	d = 0.0
	for i in range(20):
		d_old = d
		L = np.dot(P.T, R)
		B = np.dot(P, L**3 - (1.0/m)*np.dot(L, np.diag(np.diag(np.dot(L.T, L)))))
		U, s, V = np.linalg.svd(B)
		R = np.dot(U, V)
		d = np.sum(s)
		if (d_old != 0) and (d < d_old*(1.0 + 1e-6)):
			print(str(i+1) + " iterations performed")
			break
	return np.dot(P.T, R), R

# pcadapt scan
def pcadaptScan(L, Pi, f, K, t):
	n, m = Pi.shape # Dimensions
	E = np.empty((n, m), dtype=np.float32)
	Z = np.empty((m, K), dtype=np.float32)
	covariance_cy.updatePCAngsd(L, Pi, E, t)
	covariance_cy.standardizeE(E, f, t)

	# Performing SVD on normalized expected genotypes
	V, s, U = svds(E, k=K)

	# pcadapt computations
	B = np.dot(U.T, np.diag(s)) # Betas (m x K)
	shared.computeZ(E, B, V, Z)
	return Z