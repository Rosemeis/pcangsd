"""
Estimates the covariance matrix for NGS data using the PCAngsd method,
by estimating individual allele frequencies based on SVD.
"""

__author__ = "Jonas Meisner"

# Import libraries
import numpy as np
import covariance_cy
import shared
from scipy.sparse.linalg import svds, eigsh

# Estimate individual allele frequencies
def estimateSVD(E, e, Pi):
	# Reduced SVD of rank K (Scipy library)
	W, s, U = svds(E, k=e)
	np.dot(W*s, U, out=Pi)
	return W

##### PCAngsd #####
def pcaEM(L, e, f, m_iter, m_tole, t):
	n, m = L.shape # Dimension of likelihood matrix
	n //= 3 # Number of individuals
	K = e

	# Initiate matrices
	E = np.empty((n, m), dtype=np.float32)
	X = np.empty((n, m), dtype=np.float32)
	dCov = np.zeros(n, dtype=np.float32)

	# Estimate covariance matrix (Fumagalli) and infer number of PCs
	if K == 0:
		# Prepare dosages and diagonal
		covariance_cy.covFumagalli(L, f, E, dCov, t)
		covariance_cy.standardizeE(E, f, t)
		C = np.dot(E, E.T)/m
		np.fill_diagonal(C, dCov)

		if m_iter == 0:
			print("Returning with ngsTools covariance matrix!")
			return C, None, K

		# Velicer's Minimum Average Partial (MAP) Test
		eigVals, eigVecs = eigsh(C, k=min(n-1, 15)) # Eigendecomposition (Symmetric) - ARPACK
		eigVals = eigVals[::-1] # Sorted eigenvalues
		eigVals[eigVals < 0] = 0
		eigVecs = eigVecs[:, ::-1] # Sorted eigenvectors
		loadings = eigVecs*np.sqrt(eigVals)
		mapTest = np.empty(min(m-1, 15), dtype=np.float32)

		# Loop over m-1 eigenvalues for MAP test (Shriner implementation)
		for eig in range(min(m-1, 15)):
			partcov = C - (np.dot(loadings[:, 0:(eig + 1)], loadings[:, 0:(eig + 1)].T))
			d = np.diag(partcov)

			if (np.sum(np.isnan(d)) > 0) or (np.sum(d == 0) > 0) or (np.sum(d < 0) > 0):
				mapTest[eig] = 1
			else:
				d = np.diagflat(1/np.sqrt(d))
				pr = np.dot(d, np.dot(partcov, d))
				mapTest[eig] = (np.sum(pr**2) - m)/(m*(m - 1))

		K = max([1, np.argmin(mapTest) + 1]) # Number of principal components retained
		print("Using " + str(K) + " principal components (MAP test)")

		# Release memory
		del eigVals, eigVecs, loadings, partcov, mapTest

	else:
		print("Using " + str(K) + " principal components (manually selected)")
	covariance_cy.updateFumagalli(L, f, E, t)

	# Estimate individual allele frequencies
	Pi = np.empty((n, m), dtype=np.float32)
	covariance_cy.centerE(E, f, t)
	W = estimateSVD(E, K, Pi)
	covariance_cy.updatePi(Pi, f, t)
	prevW = np.copy(W)
	print("Individual allele frequencies estimated (1)")

	# Iterative estimation
	for iteration in range(2, m_iter+1):
		covariance_cy.updatePCAngsd(L, Pi, E, t)

		# Estimate individual allele frequencies
		covariance_cy.centerE(E, f, t)
		W = estimateSVD(E, K, Pi)
		covariance_cy.updatePi(Pi, f, t)

		# Break iterative update if converged
		diff = covariance_cy.rmse2d_eig(W, prevW)
		print("Individual allele frequencies estimated (" + str(iteration) + "). RMSE=" + str(diff))
		if diff < m_tole:
			print("Estimation of individual allele frequencies has converged.")
			break
		prevW = np.copy(W)
	del W, prevW

	# Estimate covariance matrix (PCAngsd)
	covariance_cy.covPCAngsd(L, f, Pi, E, dCov, t)
	covariance_cy.standardizeE(E, f, t)
	C = np.dot(E, E.T)/m
	np.fill_diagonal(C, dCov)

	return C, Pi, K