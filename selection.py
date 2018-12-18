"""
Selection scan using principal components based on Galinsky et al. (2016) and Luu et al. (2017)

Outputs the chisquare distributed selection statistics for each of the top
principal components.
"""

__author__ = "Jonas Meisner"

# Import libraries
import numpy as np
import threading
from math import sqrt
from numba import jit
from scipy.sparse.linalg import eigsh

# Normalize the posterior expectations of the genotypes
@jit("void(f4[:, :], f8[:], i8, i8, f8[:, :])", nopython=True, nogil=True, cache=True)
def normalizeGeno(E, f, S, N, X):
	m, n = E.shape
	for ind in xrange(S, min(S+N, m)):
		for s in xrange(n):
			X[ind, s] = (E[ind, s] - 2*f[s])/sqrt(2*f[s]*(1 - f[s]))

# Selection scan
def selectionScan(E, f, C, e, model=1, t=1):
	m, n = E.shape # Dimensions

	# Perform eigendecomposition on covariance matrix
	Sigma, V = eigsh(C, k=e) # Eigendecomposition (Symmetric) - ARPACK
	Sigma = Sigma[::-1] # Sorted eigenvalues
	V = V[:, ::-1] # Sorted eigenvectors

	# Multithreading parameters
	chunk_N = int(np.ceil(float(m)/t))
	chunks = [i * chunk_N for i in xrange(t)]

	if model==1: # FastPCA
		X = np.empty((m, n))

		# Multithreading
		threads = [threading.Thread(target=normalizeGeno, args=(E, f, chunk, chunk_N, X)) for chunk in chunks]
		for thread in threads:
			thread.start()
		for thread in threads:
			thread.join()

		# Test statistic container
		test = np.zeros((e, n))

		# Compute p-values for each PC in each site
		for ev in xrange(e):
			# Weighted SNPs are chi-square distributed with df = 1
			test[ev] = (np.dot(X.T, V[:, ev])**2)/Sigma[ev]


	elif model==2: # PCAdapt
		test = np.zeros(n)

		# Linear regressions
		hatX = np.dot(np.linalg.inv(np.dot(V.T, V)), V.T)
		B = np.dot(hatX, E)
		res = E - np.dot(V, np.dot(hatX, E))

		# Z-scores estimation
		resStd = np.std(res, axis=0, ddof=1) # Standard deviations of residuals
		Z = B/resStd
		Z = np.nan_to_num(Z) # Set NaNs to 0
		Zmeans = np.mean(Z, axis=1) # K mean Z-scores
		Zinvcov = np.linalg.inv(np.cov(Z)) # Inverse covariance matrix of Z-scores

		# Calculate Mahalanobis distances
		for s in xrange(n):
			test[s] = np.sqrt(np.dot(np.dot((Z[:, s] - Zmeans), Zinvcov), (Z[:, s] - Zmeans)))

	return test
