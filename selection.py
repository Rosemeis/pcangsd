"""
Selection scan using principal components based on Galinsky et al. (2016) and Luu et al. (2017)

Outputs the chisquare distributed selection statistics for each of the top
principal components.
"""

__author__ = "Jonas Meisner"

# Import libraries
import numpy as np
from math import sqrt
from numba import jit
import threading

# Normalize the posterior expectations of the genotypes
@jit("void(f4[:, :], f8[:], i8, i8, f8[:, :])", nopython=True, nogil=True, cache=True)
def normalizeGeno(expG, f, S, N, X):
	m, n = expG.shape
	for ind in xrange(S, min(S+N, m)):
		for s in xrange(n):
			X[ind, s] = (expG[ind, s] - 2*f[s])/sqrt(2*f[s]*(1 - f[s]))

# Selection scan
def selectionScan(expG, f, C, nEV, model=1, threads=1):
	# Perform eigendecomposition on covariance matrix
	m, n = expG.shape
	eigVals, eigVecs = np.linalg.eigh(C) # Eigendecomposition (Symmetric)
	sort = np.argsort(eigVals)[::-1] # Sorting vector
	l = eigVals[sort[:nEV]] # Sorted eigenvalues
	V = eigVecs[:, sort[:nEV]] # Sorted eigenvectors

	chunk_N = int(np.ceil(float(m)/threads))
	chunks = [i * chunk_N for i in xrange(threads)]

	if model==1: # FastPCA
		X = np.empty((m, n))

		# Multithreading
		threads = [threading.Thread(target=normalizeGeno, args=(expG, f, chunk, chunk_N, X)) for chunk in chunks]
		for thread in threads:
			thread.start()
		for thread in threads:
			thread.join()

		# Test statistic container
		test = np.zeros((nEV, n))

		# Compute p-values for each PC in each site
		for eigVec in xrange(nEV):
			# Weighted SNPs are chi-square distributed with df = 1
			test[eigVec] = (1.0/l[eigVec])*(np.dot(X.T, V[:, eigVec])**2)


	elif model==2: # PCAdapt
		test = np.zeros(n)

		# Linear regressions
		hatX = np.dot(np.linalg.inv(np.dot(V.T, V)), V.T)
		B = np.dot(hatX, expG)
		res = expG - np.dot(V, np.dot(hatX, expG))

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
