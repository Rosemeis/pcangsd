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
def PCAngsd(likeMatrix, EVs, M, f, M_tole=1e-4, regLR=0, scaledLR=False, LD=0):
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

	# LD regression
	if LD > 0:
		print "Performing LD regression"
		Wr = np.zeros((m, n))
		diagWr = np.zeros(m)

		# Compute the residual genotype matrix
		for s in range(1, n):
			# Setting up sites to use
			if s < LD:
				sArray = np.arange(s) # Preceding sites
			else:
				sArray = np.arange(s-LD, s) # Preceding sites

			X_LD = np.hstack((np.ones((m, 1)), X[:, sArray]))
			y_LD = X[:, s]
			Tau_LD = np.eye(X_LD.shape[1])*0.01
			hat_LD = np.dot(np.linalg.inv(np.dot(X_LD.T, X_LD) + Tau_LD), X_LD.T)
			Wr[:, s] =  np.dot(X_LD, np.dot(hat_LD, y_LD))

		for ind in range(m):
			diagWr[ind] = np.dot(Wr[ind].T, Wr[ind])

		C = np.dot(X - Wr, (X - Wr).T) # Covariance matrix for i != j
		np.fill_diagonal(C, (diagC - diagWr)) # Entries for i == j
		C = C/np.sum(np.var(X - Wr, axis=0)) # Normalize covariance matrix
		print "Covariance matrix computed	(0 - LD)"

	else:
		C = np.dot(X, X.T)/n # Covariance matrix for i != j
		np.fill_diagonal(C, diagC) # Entries for i == j
		print "Covariance matrix computed	(Fumagalli)"
	
	prevF = np.ones((m, n))*np.inf # Container for break condition
	nEV = EVs
	
	# Iterative covariance estimation
	for iteration in range(1, M+1):
		
		# Eigen-decomposition
		eigVals, eigVecs = np.linalg.eig(C*n)
		sort = np.argsort(eigVals)[::-1] # Sorting vector
		evSort = eigVals[sort] # Sorted eigenvalues

		# Patterson test for number of significant eigenvalues
		if (iteration == 1) & (EVs == 0):
			Vscale = np.zeros(10)
			#mPrime = m-1
			#nPrime = ((mPrime+1)*(np.sum(evSort[:mPrime])**2))/(((mPrime-1)*np.sum(evSort[:mPrime]**2))-(np.sum(evSort[:mPrime])**2))
			nPrime = n
			mPrime = m

			# Normalizing eigenvalues - Estimating Wilshart parameters
			mu = ((np.sqrt(nPrime-1) + np.sqrt(mPrime))**2)
			sigma = (np.sqrt(nPrime-1) + np.sqrt(mPrime))*(((1/np.sqrt(nPrime-1)) + (1/np.sqrt(mPrime)))**(1/3.0))
			#l = (mPrime*evSort[0])/np.sum(evSort[:mPrime]) # Scale eigenvalues
			x = (evSort[0] - mu)/sigma # Normalized eigenvalues (test statistic)
			Vscale[0] = x

			# Test TW statistics for significance
			if x > 2.0236:
				nEV = 1
			assert (nEV !=0), "0 significant eigenvalues found. Select number of eigenvalues manually!"

			while True:
				#mPrime = m-1-nEV
				#nPrime = ((mPrime+1)*(np.sum(evSort[nEV:mPrime])**2))/(((mPrime-1)*np.sum(evSort[nEV:mPrime]**2))-(np.sum(evSort[nEV:mPrime])**2))
				nPrime = n
				mPrime = m-nEV
				
				# Normalizing eigenvalues - Estimating Wilshart parameters
				mu = ((np.sqrt(nPrime-1) + np.sqrt(mPrime))**2)
				sigma = (np.sqrt(nPrime-1) + np.sqrt(mPrime))*(((1/np.sqrt(nPrime-1)) + (1/np.sqrt(mPrime)))**(1/3.0))
				#l = (mPrime*evSort[nEV])/np.sum(evSort[nEV:mPrime])
				x = (evSort[nEV] - mu)/sigma
				Vscale[nEV] = x

				# Test TW statistics for significance
				if x > 2.0236:
					nEV += 1
					if nEV == 10:
					 	print str(nEV) + " eigenvalue(s) are significant"
					 	Vscale = Vscale/Vscale[0]
					 	break
				else:
					print str(nEV) + " eigenvalue(s) are significant"
					Vscale = Vscale/Vscale[0]
					break


		V = eigVecs[:, sort[:nEV]] # Sorted eigenvectors regarding eigenvalue size

		# Multiple linear regression
		V_bias = np.hstack((np.ones((m, 1)), V)) # Add bias term

		if scaledLR: # Scaled regression
			V_bias = V_bias*np.append(np.array([1]), Vscale)

		if regLR > 0: # Ridge regression
			Tau = np.eye(V_bias.shape[1])*regLR
			hatX = np.dot(np.linalg.inv(np.dot(V_bias.T, V_bias) + Tau), V_bias.T)
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

		# LD regression
		if LD > 0:
			print "Performing LD regression"
			Wr = np.zeros((m, n))
			diagWr = np.zeros(m)

			# Compute the residual genotype matrix
			for s in range(1, n):
				# Setting up sites to use
				if s < LD:
					sArray = np.arange(s) # Preceding sites
				else:
					sArray = np.arange(s-LD, s) # Preceding sites

				X_LD = X[:, sArray]
				y_LD = X[:, s]
				Tau_LD = np.eye(X_LD.shape[1])*0.01
				hat_LD = np.dot(np.linalg.inv(np.dot(X_LD.T, X_LD) + Tau_LD), X_LD.T)
				Wr[:, s] =  np.dot(X_LD, np.dot(hat_LD, y_LD))

			for ind in range(m):
				diagWr[ind] = np.dot(Wr[ind].T, Wr[ind])

			C = np.dot(X - Wr, (X - Wr).T) # Covariance matrix for i != j
			np.fill_diagonal(C, (diagC - diagWr)) # Entries for i == j
			C = C/np.sum(np.var(X - Wr, axis=0)) # Normalize covariance matrix

		else:
			C = np.dot(X, X.T)/n # Covariance matrix for i != j
			np.fill_diagonal(C, diagC) # Entries for i == j

		# Break iterative covariance update if converged
		updateDiff = rmse(predF, prevF)
		print "Covariance matrix computed	(" + str(iteration) + ") Diff=" + str(updateDiff)
		if updateDiff <= M_tole:
			print "PCAngsd converged at iteration: " + str(iteration)
			break

		prevF = np.copy(predF) # Update break condition

	return C, predF, nEV, X, expG
