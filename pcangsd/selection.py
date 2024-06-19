"""
PCAngsd.
Perform PC-based selection scans or extract SNP weights.
"""

__author__ = "Jonas Meisner"

# Libraries
import numpy as np
from scipy.sparse.linalg import svds

# Import scripts
from pcangsd import shared_cy
from pcangsd import covariance_cy

##### Selection scans #####
# FastPCA - Galinsky et al.
def galinskyScan(L, P, f, K, t):
	m, n = P.shape
	E = np.zeros((m, n), dtype=np.float32)
	D = np.zeros((m, K), dtype=np.float32)
	covariance_cy.updateSelection(L, P, E, f, t)

	# Perform SVD on standardized posterior expectations
	U, _, _ = svds(E, k=K)
	U = U[:,::-1]
	shared_cy.computeD(U, D)
	del E, U
	return D

# pcadapt
def pcadaptScan(L, P, f, K, t):
	m, n = P.shape
	E = np.zeros((m, n), dtype=np.float32)
	Z = np.zeros((m, K), dtype=np.float32)
	covariance_cy.updateSelection(L, P, E, f, t)

	# Perform SVD on standardized posterior expectations
	U, s, Vt = svds(E, k=K)
	B = np.dot(U, np.diag(s)) # Betas (m x K)
	shared_cy.computeZ(E, B, Vt, Z)
	del B, E, U, s, Vt
	return Z

# SNP weights
def snpWeights(L, P, f, K, t):
	m, n = P.shape
	E = np.zeros((m, n), dtype=np.float32)
	covariance_cy.updateSelection(L, P, E, f, t)

	# Perform SVD on standardized posterior expectations
	U, s, _ = svds(E, k=K)
	snpW = U[:,::-1]*(s[::-1]**2)/float(m) # Loadings
	return snpW
