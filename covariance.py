"""
Estimates the covariance matrix for NGS data using the PCAngsd method,
by linear modelling of expected genotypes based on principal components.
"""

__author__ = "Jonas Meisner"

# Import functions
from helpFunctions import *
from emMAF import *

# Import libraries
import numpy as np

# PCAngsd 
def PCAngsd(likeMatrix, EVs, M, EM, M_tole=1e-5, EM_tole=1e-6, lrReg=False):
	mTotal, n = likeMatrix.shape # Dimension of likelihood matrix
	m = mTotal/3 # Number of individuals

	# Estimate average allele frequencies
	f = alleleEM(likeMatrix, EM, EM_tole)
	mask = (f >= 0.05) & (f <= 0.95) # Only select variable sites
	f = f[mask] # Variable sites
	fMatrix = np.vstack(((1-f)**2, 2*f*(1-f), f**2)) # Estimated genotype frequencies under HWE
	nSites = np.sum(mask) # Number of variable sites
	print "Number of sites evaluated: " + str(nSites)

	# Estimate covariance matrix
	gVector = np.array([0,1,2]) # Genotype vector
	normV = np.sqrt(2*f*(1-f)) # Normalizer for genotype matrix
	diagC = np.zeros(m) # Diagonal of covariance matrix
	expG = np.zeros((m, nSites)) # Expected genotype matrix

	for ind in range(m):
		wLike = likeMatrix[(3*ind):(3*ind+3), mask]*fMatrix # Weighted likelihoods
		gProp = wLike/np.sum(wLike, axis=0) # Genotype probabilities of individual
		gProp = np.nan_to_num(gProp) # Set NaNs to 0
		expG[ind] = np.sum((gProp.T*gVector).T, axis=0) # Expected genotypes

		# Estimate diagonal entries in covariance matrix
		diagC[ind] = np.sum(np.sum((((gVector*np.ones((nSites, 3))).T - 2*f)*gProp)**2, axis=0)/(normV**2))/nSites

	C = np.dot((expG - 2*f)/normV, ((expG - 2*f)/normV).T)/nSites # Covariance matrix for i != j
	np.fill_diagonal(C, diagC) # Entries for i == j
	print "Covariance matrix computed	(Fumagalli)"
	
	prevEG = np.ones((m, nSites))*np.inf # Container for break condition
	nEV = EVs
	
	# Iterative covariance estimation
	for iteration in range(1, M+1):
		
		# Eigen-decomposition
		eigVals, eigVecs = np.linalg.eig(C)
		sort = np.argsort(eigVals)[::-1] # Sorting vector
		evSort = eigVals[sort][:-1] # Sorted eigenvalues

		# Patterson test for number of significant eigenvalues
		if iteration==1 and EVs==0:

			for ev in range(10): # Loop over maximum 10 eigenvalues
				mPrime = m-1-ev # Rank of matrix

				# Effective number of samples
				nPrime = ((mPrime+1)*(np.sum(evSort[ev:])**2))/(((mPrime-1)*np.sum(evSort[ev:]**2))-(np.sum(evSort[ev:])**2))

				# Normalizing largest eigenvalue
				mu = ((np.sqrt(nPrime-1) + np.sqrt(mPrime))**2)/nPrime
				sigma = ((np.sqrt(nPrime-1)+np.sqrt(mPrime))/nPrime)*(((1.0/np.sqrt(nPrime-1))+(1.0/np.sqrt(mPrime)))**(1.0/3.0))
				l = (mPrime*evSort[ev])/np.sum(evSort[ev:]) 
				x = (l - mu)/sigma
			
				# Test TW statistics for significance
				if x <= 0.9794:
					nEV = ev # Number of significant eigenvalues at the 0.05 signficance level
					print str(ev) + " eigenvalue(s) are significant"
					break
				elif ev == 9:
					nEV = 10
					print "10 eigenvalue(s) are significant (maximum)"
				else:
					nEV = ev+1

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
			
			wLike = likeMatrix[(3*ind):(3*ind+3), mask]*fMatrix # Weighted likelihoods
			gProp = wLike/np.sum(wLike, axis=0) # Genotype probabilities of individual
			gProp = np.nan_to_num(gProp) # Set NaNs to 0
			expG[ind] = np.sum((gProp.T*gVector).T, axis=0) # Expected genotypes

			# Estimate diagonal entries in covariance matrix
			diagC[ind] = np.sum(np.sum((((gVector*np.ones((nSites, 3))).T - 2*f)*gProp)**2, axis=0)/(normV**2))/nSites

		C = np.dot((expG - 2*f)/normV, ((expG - 2*f)/normV).T)/nSites # Covariance matrix for i != j
		np.fill_diagonal(C, diagC) # Entries for i == j

		# Break iterative covariance update if converged
		updateDiff = rmse(predEG, prevEG)
		print "Covariance matrix computed	(" +str(iteration) + ") Diff=" + str(updateDiff)
		if updateDiff <= M_tole:
			print "PCAngsd converged at iteration: " + str(iteration)
			break

		prevEG = np.copy(predEG) # Update break condition

	X = (expG - 2*f)/normV
	return C, f, predF, nEV, mask, expG, X