import numpy as np
from scipy.sparse.linalg import svds
from pcangsd import covariance_cy
from pcangsd import shared_cy

##### Selection scans #####
# FastPCA - Galinsky et al.
def galinskyScan(L, P, f, K):
	M, N = P.shape
	E = np.zeros((M, N), dtype=np.float32)
	D = np.zeros((M, K), dtype=np.float32)
	covariance_cy.updateSelection(L, P, E, f)

	# Perform SVD on standardized posterior expectations
	U, _, _ = svds(E, k=K)
	U = np.ascontiguousarray(U[:,::-1])
	shared_cy.computeD(U, D)
	return D

# pcadapt
def pcadaptScan(L, P, f, K):
	M, N = P.shape
	E = np.zeros((M, N), dtype=np.float32)
	Z = np.zeros((M, K), dtype=np.float32)
	covariance_cy.updateSelection(L, P, E, f)

	# Perform SVD on standardized posterior expectations
	U, S, Vt = svds(E, k=K)
	B = np.dot(U, np.diag(S))
	shared_cy.computeZ(E, B, np.ascontiguousarray(Vt.T), Z)
	return Z

# SNP weights
def snpWeights(L, P, f, K):
	M, N = P.shape
	E = np.zeros((M, N), dtype=np.float32)
	covariance_cy.updateSelection(L, P, E, f)

	# Perform SVD on standardized posterior expectations
	U, S, _ = svds(E, k=K)
	snpW = U[:,::-1]*(S[::-1]**2)*(1.0/float(M))
	return snpW
