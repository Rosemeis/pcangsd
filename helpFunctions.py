"""
Help functions to use in the PCAngsd framework.
"""

__author__ = "Jonas Meisner"

# Import libraries
import numpy as np
import pandas as pd


### Help functions

# Root mean squared error
def rmse(m1, m2):
	return np.sqrt(np.mean((m1 - m2)**2))


# Compute bias
def bias(mEst, mTrue):
	return np.mean(mEst-mTrue)


# Linear least squares model with Tikhonov regularization for LD regression
def linRegLD(X, y, reg=True):
	if reg:
		Tau = np.eye(X.shape[1])*0.1
		B = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X) + np.dot(Tau.T, Tau)), X.T), y)
	else:
		B = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), y)
		
	return B