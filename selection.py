"""
Selection scan using principal components based on Galinsky et al. (2016) and Luu et al. (2017)

Outputs the chisquare distributed selection statistics for each of the top
principal components.
"""

__author__ = "Jonas Meisner"

# Import libraries
import numpy as np
from helpFunctions import *

# Selection scan
def selectionScan(X, C, nEV, model=1):
	# Perform eigendecomposition on covariance matrix
	eVals, eVecs = np.linalg.eig(C)
	sort = np.argsort(eVals)[::-1]
	evSort = eVals[sort][:-1]
	V = eVecs[:, sort[:nEV]]
	l = evSort[:nEV]

	if model==1: # FastPCA
		test = np.zeros((nEV, X.shape[1]))

		# Compute p-values for each PC in each site
		for eigVec in range(nEV):
			# Weighted SNP is chi-square distributed with df = 1
			test[eigVec] = (1.0/l[eigVec])*(np.dot(X.T, V[:, eigVec])**2)


	elif model==2: # PCAdapt
		test = np.zeros(X.shape[1])

		V_bias = np.hstack((np.ones((X.shape[0], 1)), V)) # Add bias term

		# Linear regressions
		hatX = np.dot(np.linalg.inv(np.dot(V_bias.T, V_bias)), V_bias.T)
		B = np.dot(hatX, X)
		res = X - np.dot(V_bias, np.dot(hatX, X))

		# Z-scores estimation
		resStd = np.std(res, axis=0, ddof=1) # Standard deviations of residuals
		Z = B[1:]/resStd
		Z = np.nan_to_num(Z) # Set NaNs to 0
		Zmeans = np.mean(Z, axis=1) # K mean Z-scores
		Zinvcov = np.linalg.inv(np.cov(Z)) # Inverse covariance matrix of Z-scores

		# Calculate Mahalanobis distances
		for s in range(X.shape[1]):
			test[s] = np.sqrt(np.dot(np.dot((Z[:, s] - Zmeans), Zinvcov), (Z[:, s] - Zmeans)))

	return test