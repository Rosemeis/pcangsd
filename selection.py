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
def selectionScan(X, C, nEV):
	chisqVec = np.zeros((nEV, X.shape[1]))

	# Perform eigendecomposition on covariance matrix
	eVals, eVecs = np.linalg.eig(C)
	sort = np.argsort(eVals)[::-1]
	evSort = eVals[sort][:-1]
	V = eVecs[:, sort[:nEV]]
	l = evSort[:nEV]

	# Compute p-values for each PC in each site
	for eigVec in range(nEV):
		# Weighted SNP is chi-square distributed with df = 1
		chisqVec[eigVec] = (1.0/l[eigVec])*(np.dot(X.T, V[:, eigVec])**2)

	return chisqVec