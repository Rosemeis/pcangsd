"""
Helper functions to use in the PCAngsd framework.
"""

__author__ = "Jonas Meisner"

# Import libraries
import numpy as np
import pandas as pd
from scipy.stats import binom, poisson


### Help functions

# Root mean squared error
def rmse(m1, m2):
	return np.sqrt(np.mean((m1 - m2)**2))


# Compute bias
def bias(mEst, mTrue):
	return np.mean(mEst-mTrue)


# Simulate genotype likelihoods (diallelic)
def genoLikes(x, d=5, e=0.01, norm=False):
	n = len(x) # Number of SNPs
	depth = np.random.poisson(d,n) # Depth vector
	nA = np.random.binomial(depth, np.array([e, 0.5, 1-e])[x], n) # Number of ancestral alleles

	# Generate likelihoods
	like = np.vstack((binom.pmf(nA,depth,e), binom.pmf(nA,depth,0.5), binom.pmf(nA,depth,1-e)))

	# Normalize likelihoods
	if norm:
		like = like/np.sum(like, axis=0)

	return like


# Linear least squares model with Tikhonov regularizaton (Ridge regression)
def linReg(X, y, reg=True):
	if reg:
		Tau = np.eye(X.shape[1])*0.1*np.arange(X.shape[1])
		B = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X) + np.dot(Tau.T, Tau)), X.T), y)
	else:
		B = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), y)
		
	return B


# Linear least squares model with Tikhonov regularization for LD regression
def linRegLD(X, y, reg=True):
	if reg:
		Tau = np.eye(X.shape[1])*0.01
		B = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X) + np.dot(Tau.T, Tau)), X.T), y)
	else:
		B = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), y)
		
	return B
