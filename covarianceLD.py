"""
Estimates the covariance matrix for NGS data using the PCAngsd method,
by linear modelling of expected genotypes based on principal components.

This function includes LD regression.
"""

__author__ = "Jonas Meisner"

# Import functions
from helpFunctions import *
from emMAF import *

# Import libraries
import numpy as np
import pandas as pd

# PCAngsd 
def PCAngsdLD(likeMatrix, LD, EVs, M, f, M_tole=1e-4, lrReg=False):
	mTotal, n = likeMatrix.shape # Dimension of likelihood matrix
	m = mTotal/3 # Number of individuals

	# Estimate covariance matrix
	fMatrix = np.vstack(((1-f)**2, 2*f*(1-f), f**2)) # Estimated genotype frequencies under HWE
	gVector = np.array([0,1,2]).reshape(3, 1) # Genotype vector
	normV = np.sqrt(2*f*(1-f)) # Normalizer for genotype matrix
	diagC = np.zeros(m) # Diagonal of covariance matrix
	expG = np.zeros((m, n)) # Expected genotype matrix

	for ind in range(m):
		wLike = likeMatrix[(3*ind):(3*ind+3)]*fMatrix # Weighted likelihoods
		gProp = wLike/np.sum(wLike, axis=0) # Genotype probabilities of individual
		expG[ind] = np.sum(gProp*gVector, axis=0) # Expected genotypes

		# Estimate diagonal entries in covariance matrix
		diagC[ind] = np.sum(np.sum(((np.ones((3, n))*gVector - 2*f)*gProp)**2, axis=0)/(normV**2))/n

	X = (expG - 2*f)/normV # Standardized genotype matrix
	C = np.dot(X, X.T)/n # Covariance matrix for i != j
	np.fill_diagonal(C, diagC) # Entries for i == j
	print "Covariance matrix computed	(Fumagalli)"
	
	prevEG = np.ones((m, n))*np.inf # Container for break condition
	nEV = EVs
	
	# Iterative covariance estimation
	for iteration in range(1, M+1):
		
		# Eigen-decomposition
		eigVals, eigVecs = np.linalg.eig(C)
		sort = np.argsort(eigVals)[::-1] # Sorting vector
		evSort = eigVals[sort][:-1] # Sorted eigenvalues

		# Patterson test for number of significant eigenvalues
		if iteration==1 and EVs==0:
			mPrime = m-1
			nPrime = ((mPrime+1)*(np.sum(evSort)**2))/(((mPrime-1)*np.sum(evSort**2))-(np.sum(evSort)**2))

			# Normalizing eigenvalues
			mu = ((np.sqrt(nPrime-1) + np.sqrt(mPrime))**2)/nPrime
			sigma = np.sqrt(np.sum(evSort))/((mPrime-1)*nPrime)
			l = (mPrime*evSort)/np.sum(evSort) 
			x = (l - mu)/sigma

			# Test TW statistics for significance
			nEV = np.sum(x > 0.9794)

			if nEV > 1:
				print str(nEV) + " eigenvalues are significant"
			else:
				# Testing for additional structure
				mPrime = m-2
				nPrime = ((mPrime+1)*(np.sum(evSort[1:])**2))/(((mPrime-1)*np.sum(evSort[1:]**2))-(np.sum(evSort[1:])**2))

				# Normalizing eigenvalues
				mu = ((np.sqrt(nPrime-1) + np.sqrt(mPrime))**2)/nPrime
				sigma = np.sqrt(np.sum(evSort[1:]))/((mPrime-1)*nPrime)
				l = (mPrime*evSort)/np.sum(evSort[1:]) 
				x = (l - mu)/sigma

				# Test TW statistics for significance
				addEV = np.sum(x > 0.9794)

				if (nEV + addEV) > 10:
					nEV = 10
					print str(nEV) + " eigenvalues are significant (maximum)"
				else:
					nEV = nEV + addEV
					print str(nEV) + " eigenvalue(s) are significant"

			assert (nEV !=0), "0 significant eigenvalues found. Select number of eigenvalues manually!"


		V = eigVecs[:, sort[:nEV]] # Sorted eigenvectors regarding eigenvalue size

		# Multiple linear regression
		V_bias = np.hstack((np.ones((m, 1)), V)) # Add bias term

		if lrReg:
			Tau = np.eye(V_bias.shape[1])*0.1*np.arange(V_bias.shape[1])
			hatX = np.dot(np.linalg.inv(np.dot(V_bias.T, V_bias) + np.dot(Tau.T, Tau)), V_bias.T)
		else:
			hatX = np.dot(np.linalg.inv(np.dot(V_bias.T, V_bias)), V_bias.T)

		predEG = np.dot(V_bias, np.dot(hatX, expG))
		predF = predEG/2 # Estimated allele frequencies from expected genotypes
		predF = predF.clip(min=0.00001, max=1-0.00001)
		
		# Estimate covariance matrix
		for ind in range(m):
			# Genotype frequencies based on individual allele frequencies under HWE 
			fMatrix = np.vstack(((1-predF[ind])**2, 2*predF[ind]*(1-predF[ind]), predF[ind]**2))
			
			wLike = likeMatrix[(3*ind):(3*ind+3)]*fMatrix # Weighted likelihoods
			gProp = wLike/np.sum(wLike, axis=0) # Genotype probabilities of individual
			expG[ind] = np.sum(gProp*gVector, axis=0) # Expected genotypes

			# Estimate diagonal entries in covariance matrix
			diagC[ind] = np.sum(np.sum(((np.ones((3, n))*gVector - 2*f)*gProp)**2, axis=0)/(normV**2))/n

		X = (expG - 2*f)/normV # Standardized genotype matrix
		C = np.dot(X, X.T)/n # Covariance matrix for i != j
		np.fill_diagonal(C, diagC) # Entries for i == j

		# Break iterative covariance update if converged
		updateDiff = rmse(predEG, prevEG)
		print "Covariance matrix computed	(" +str(iteration) + ") Diff=" + str(updateDiff)
		if updateDiff <= M_tole:
			print "PCAngsd converged at iteration: " + str(iteration)
			break

		prevEG = np.copy(predEG) # Update break condition

	
	# LD regression
	print "Performing LD regression"

	# Setting up for LD
	Wr = np.zeros((m, n))
	diagWr = np.zeros(m)

	# Compute the residual genotype matrix R
	for site in range(n):
		# Setting up sites to use
		if site == 0:
			s1 = np.array([], dtype=int) # Preceding sites
			s2 = np.arange(site+1, site+1+LD) # Following sites
			sArray = np.append(s1, s2)
		elif site < LD:
			s1 = np.arange(site) # Preceding sites
			s2 = np.arange(site+1, site+1+LD) # Following sites
			sArray = np.append(s1, s2)
		elif site == n:
			s1 = np.arange(site-LD, site) # Preceding sites
			s2 = np.array([], dtype=int) # Following sites
			sArray = np.append(s1, s2)
		elif site + LD >= n:
			s1 = np.arange(site-LD, site) # Preceding sites
			s2 = np.arange(site+1, n) # Following sites
			sArray = np.append(s1, s2)
		else:
			s1 = np.arange(site-LD, site) # Preceding sites
			s2 = np.arange(site+1, site+1+LD) # Following sites
			sArray = np.append(s1, s2)

		B = linRegLD(X[:, sArray], X[:, site], True)
		Wr[:, site] =  np.dot(X[:, sArray], B)

	for ind in range(m):
		diagWr[ind] = np.dot(Wr[ind].T, Wr[ind])

	R = X - Wr
	C = np.dot(R, R.T) # Covariance matrix for i != j
	np.fill_diagonal(C, (diagC - diagWr)) # Entries for i == j
	C = C/np.sum(np.var(R, axis=0)) # Normalize covariance matrix

	return C, predF, nEV, X, expG