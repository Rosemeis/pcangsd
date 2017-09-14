"""
Help functions to use for chunk estimations in the PCAngsd framework.
"""

__author__ = "Jonas Meisner"

# Import libraries
import numpy as np
import pandas as pd

# Root mean squared error
def rmse(m1, m2):
	return np.sqrt(np.mean((m1 - m2)**2))


# Estimate posterior expectations of the genotypes
def chunkEstimation(likeMatrix, f):
	mTotal, n = likeMatrix.shape # Dimension of likelihood matrix
	m = mTotal/3 # Number of individuals
	gVector = np.array([0,1,2]).reshape(3, 1) # Genotype vector
	expG = np.zeros((m, n)) # Expected genotype matrix

	if f.ndim == 1:
		fMatrix = np.vstack(((1-f)**2, 2*f*(1-f), f**2)) # Estimated genotype frequencies under HWE

	for ind in range(m):
		if f.ndim == 2:
			# Genotype frequencies based on individual allele frequencies under HWE
			fMatrix = np.vstack(((1-f[ind])**2, 2*f[ind]*(1-f[ind]), f[ind]**2))
		
		wLike = likeMatrix[(3*ind):(3*ind+3)]*fMatrix # Weighted likelihoods
		gProp = wLike/np.sum(wLike, axis=0) # Genotype probabilities of individual
		expG[ind] = np.sum(gProp*gVector, axis=0) # Expected genotypes

	return expG


# Estimate individual allele frequencies
def individualF(likeMatrix, f, C, nEV, M, M_tole, regLR, scaledLR):
	mTotal, n = likeMatrix.shape # Dimension of likelihood matrix
	m = mTotal/3 # Number of individuals
	normV = np.sqrt(2*f*(1-f)) # Normalizer for genotype matrix
	prevF = np.ones((m, n))*np.inf # Container for break condition
	predF = f # Set up for first iteration

	# Eigen-decomposition
	eigVals, eigVecs = np.linalg.eig(C)
	sort = np.argsort(eigVals)[::-1] # Sorting vector
	evSort = eigVals[sort] # Sorted eigenvalues
	V = eigVecs[:, sort[:nEV]] # Sorted eigenvectors regarding eigenvalue size
	V_bias = np.hstack((np.ones((m, 1)), V)) # Add bias term

	# Multiple Linear Regression
	if scaledLR: # Scaled regression
		V_bias = V_bias * np.append(np.array([1]), evSort[:nEV]/np.sum(evSort[:nEV]))

	if regLR > 0: # Ridge regression
		Tau = np.eye(V_bias.shape[1])*regLR
		hatX = np.dot(np.linalg.inv(np.dot(V_bias.T, V_bias) + Tau), V_bias.T)
	else:
		hatX = np.dot(np.linalg.inv(np.dot(V_bias.T, V_bias)), V_bias.T)

	for iteration in range(M):
		expG = chunkEstimation(likeMatrix, predF)		
		predEG = np.dot(V_bias, np.dot(hatX, expG))
		predF = predEG/2 # Estimated allele frequencies from expected genotypes
		predF = predF.clip(min=0.00001, max=1-0.00001)

		# Break iterative covariance update if converged
		updateDiff = rmse(predF, prevF)
		if updateDiff <= M_tole:
			break

		prevF = np.copy(predF) # Update break condition

	return predF, expG