"""
Selection scan using principal components based on Galinsky et al. (2016)

Outputs the chisquare distributed selection statistics for each of the top
principal components.
"""

__author__ = "Jonas Meisner"

# Import libraries
import numpy as np
import scipy.stats as stats
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

			# Genomic inflation factor
			inflate = np.median(test[eigVec])/stats.chi2.median(df=1)
			test[eigVec] = test[eigVec]/inflate


	elif model==2: # PCAdapt
		test = np.zeros(X.shape[1])

		# Linear regressions
		B = np.dot(X.T, V)
		res = X - np.dot(np.dot(V, V.T), X)

		# Z-scores estimation
		resVar = np.var(res, axis=0) # Variance of residuals
		Z = B*np.sqrt(1/resVar).reshape(B.shape[0], 1)
		Zmeans = np.mean(Z, axis=0) # K mean Z-scores
		Zinvcov = np.linalg.inv(np.cov(Z.T)) # Inverse covariance matrix of Z-scores

		# Calculate Mahalanobis distances
		for s in range(X.shape[1]):
			test[s] = np.dot(np.dot((Z[s, :] - Zmeans), Zinvcov), (Z[s, :] - Zmeans))

		# Genomic inflation factor
		inflate = np.median(test)/stats.chi2.median(df=nEV)
		test = test/inflate

	return test