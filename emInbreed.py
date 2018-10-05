"""
EM algorithm to estimate individual inbreeding coefficients for NGS data using genotype likelihoods
and pre-computed allele frequencies (both sample average or individual).

Both the maximum likelihood estimator and simple estimator of the inbreeding coefficients can be computed.
The estimators can be selected by the model parameter (model=1 for MLE, model=2 for Simple).
"""

__author__ = "Jonas Meisner"

# Import help functions
from helpFunctions import rmse1d

# Import libraries
import numpy as np
import threading
from numba import jit

# Inner update - model 1
@jit("void(f4[:, :], f4[:, :], i8, i8, f8[:])", nopython=True, nogil=True, cache=True)
def inbreedEM_inner1(likeMatrix, indF, S, N, F):
	m, n = indF.shape # Dimensions
	probZ = np.empty((2, n))
	probMatrix = np.empty((3, n))

	for ind in xrange(S, min(S+N, m)):
		# Estimate posterior probabilities (Z)
		for s in xrange(n):
			# Z = 0
			probMatrix[0, s] = likeMatrix[3*ind, s]*(1 - indF[ind, s])*(1 - indF[ind, s])
			probMatrix[1, s] = likeMatrix[3*ind+1, s]*2*indF[ind, s]*(1 - indF[ind, s])
			probMatrix[2, s] = likeMatrix[3*ind+2, s]*indF[ind, s]*indF[ind, s]
			probZ[0, s] = np.sum(probMatrix[:, s])*(1 - F[ind])

			# Z = 1
			probMatrix[0, s] = likeMatrix[3*ind, s]*(1 - indF[ind, s])
			probMatrix[1, s] = 0
			probMatrix[2, s] = likeMatrix[3*ind+2, s]*indF[ind, s]
			probZ[1, s] = np.sum(probMatrix[:, s])*F[ind]
		probZ /= np.sum(probZ, axis=0)

		# Update the inbreeding coefficient
		F[ind] = np.sum(probZ[1, :])/n

# Inner update - model 2
@jit("void(f4[:, :], f4[:, :], i8, i8, f8[:])", nopython=True, nogil=True, cache=True)
def inbreedEM_inner2(likeMatrix, indF, S, N, F):
	m, n = indF.shape # Dimensions
	probVec = np.empty(3) # Container for posterior probabilities
	priorVec = np.empty(3) # Container for prior probabilities

	for ind in xrange(S, min(S+N, m)):
		expH = 0
		tempVec = np.empty(3)

		for s in xrange(n):
			# Normalize priors
			Fadj = (1 - indF[ind, s])*indF[ind, s]*F[ind]
			priorVec[0] = max(1e-4, (1 - indF[ind, s])*(1 - indF[ind, s]) + Fadj)
			priorVec[1] = max(1e-4, 2*indF[ind, s]*(1 - indF[ind, s]) - 2*Fadj)
			priorVec[2] = max(1e-4, indF[ind, s]*indF[ind, s] + Fadj)
			priorVec /= np.sum(priorVec)

			# Estimate posterior probabilities
			tempVec[0] = likeMatrix[3*ind, s]*priorVec[0]
			tempVec[1] = likeMatrix[3*ind + 1, s]*priorVec[1]
			tempVec[2] = likeMatrix[3*ind + 2, s]*priorVec[2]
			tempVec /= np.sum(tempVec)

			# ANGSD procedure
			probVec += tempVec

			# Counts of heterozygotes
			expH += 2*indF[ind, s]*(1 - indF[ind, s])
		
		probVec /= n
		probVec[0] = max(1e-4, probVec[0])
		probVec[1] = max(1e-4, probVec[1])
		probVec[2] = max(1e-4, probVec[2])
		probVec /= np.sum(probVec)
		
		# Update the inbreeding coefficient
		F[ind] = 1 - (n*probVec[1]/expH)
		F[s] = max(-1.0, F[s])
		F[s] = min(1.0, F[s])

# Population allele frequencies version (-iter 0) - Inner update - model 1
@jit("void(f4[:, :], f8[:], i8, i8, f8[:])", nopython=True, nogil=True, cache=True)
def inbreedEM_inner1_noIndF(likeMatrix, f, S, N, F):
	m, n = likeMatrix.shape # Dimension of likelihood matrix
	m /= 3 # Number of individuals
	probZ = np.empty((2, n))
	probMatrix = np.empty((3, n))

	for ind in xrange(S, min(S+N, m)):
		# Estimate posterior probabilities (Z)
		for s in xrange(n):
			# Z = 0
			probMatrix[0, s] = likeMatrix[3*ind, s]*(1 - f[s])*(1 - f[s])
			probMatrix[1, s] = likeMatrix[3*ind+1, s]*2*f[s]*(1 - f[s])
			probMatrix[2, s] = likeMatrix[3*ind+2, s]*f[s]*f[s]
			probZ[0, s] = np.sum(probMatrix[:, s])*(1 - F[ind])

			# Z = 1
			probMatrix[0, s] = likeMatrix[3*ind, s]*(1 - f[s])
			probMatrix[1, s] = 0
			probMatrix[2, s] = likeMatrix[3*ind+2, s]*f[s]
			probZ[1, s] = np.sum(probMatrix[:, s])*F[ind]
		probZ /= np.sum(probZ, axis=0)

		# Update the inbreeding coefficient
		F[ind] = np.sum(probZ[1, :])/n

# Population allele frequencies version (-iter 0) - Inner update - model 2
@jit("void(f4[:, :], f8[:], i8, i8, f8[:])", nopython=True, nogil=True, cache=True)
def inbreedEM_inner2_noIndF(likeMatrix, f, S, N, F):
	m, n = likeMatrix.shape # Dimension of likelihood matrix
	m /= 3 # Number of individuals
	probVec = np.empty(3) # Container for posterior probabilities
	priorVec = np.empty(3) # Container for prior probabilities
	expH = np.sum(2*f*(1 - f)) # Expected number of heterozygotes

	for ind in xrange(S, min(S+N, m)):
		tempVec = np.empty(3)

		for s in xrange(n):
			# Normalize priors
			Fadj = (1 - f[s])*f[s]*F[ind]
			priorVec[0] = max(1e-4, (1 - f[s])*(1 - f[s]) + Fadj)
			priorVec[1] = max(1e-4, 2*f[s]*(1 - f[s]) - 2*Fadj)
			priorVec[2] = max(1e-4, f[s]*f[s] + Fadj)
			priorVec /= np.sum(priorVec)

			# Estimate posterior probabilities
			tempVec[0] = likeMatrix[3*ind, s]*priorVec[0]
			tempVec[1] = likeMatrix[3*ind + 1, s]*priorVec[1]
			tempVec[2] = likeMatrix[3*ind + 2, s]*priorVec[2]
			tempVec /= np.sum(tempVec)

			# ANGSD procedure
			probVec += tempVec
		
		probVec /= n
		probVec[0] = max(1e-4, probVec[0])
		probVec[1] = max(1e-4, probVec[1])
		probVec[2] = max(1e-4, probVec[2])
		probVec /= np.sum(probVec)
		
		# Update the inbreeding coefficient
		F[ind] = 1 - (n*probVec[1]/expH)
		F[s] = max(-1.0, F[s])
		F[s] = min(1.0, F[s])


# EM algorithm for estimation of inbreeding coefficients
def inbreedEM(likeMatrix, indF, model, EM=200, EM_tole=1e-4, t=1):
	m, n = likeMatrix.shape # Dimension of genotype likelihood matrix
	m /= 3 # Number of individuals
	F = np.ones(m)*0.25 # Initialization of inbreeding coefficients
	F_prev = np.copy(F)

	# Multithreading parameters
	chunk_N = int(np.ceil(float(m)/t))
	chunks = [i * chunk_N for i in xrange(t)]

	# Model 1 - (Hall et al.)
	if model == 1:
		for iteration in xrange(1, EM + 1):
			if indF.ndim == 2:
				# Multithreading - Update F
				threads = [threading.Thread(target=inbreedEM_inner1, args=(likeMatrix, indF, chunk, chunk_N, F)) for chunk in chunks]
				for thread in threads:
					thread.start()
				for thread in threads:
					thread.join()
			else:
				# Population allele frequencies version (-iter 0) - Multithreading - Update F
				threads = [threading.Thread(target=inbreedEM_inner1_noIndF, args=(likeMatrix, indF, chunk, chunk_N, F)) for chunk in chunks]
				for thread in threads:
					thread.start()
				for thread in threads:
					thread.join()

			# Break EM update if converged
			updateDiff = rmse1d(F, F_prev)
			print "Inbreeding coefficients estimated (" + str(iteration) + "). RMSD=" + str(updateDiff)
			if updateDiff < EM_tole:
				print "EM (Inbreeding - sites) converged at iteration: " + str(iteration)
				break
			F_prev = np.copy(F)

	# Model 2 - (Vieira et al.)
	if model == 2:
		for iteration in xrange(1, EM + 1):
			if indF.ndim == 2:
				# Multithreading - Update F
				threads = [threading.Thread(target=inbreedEM_inner2, args=(likeMatrix, indF, chunk, chunk_N, F)) for chunk in chunks]
				for thread in threads:
					thread.start()
				for thread in threads:
					thread.join()
			else:
				# Population allele frequencies version (-iter 0) - Multithreading - Update F
				threads = [threading.Thread(target=inbreedEM_inner2_noIndF, args=(likeMatrix, indF, chunk, chunk_N, F)) for chunk in chunks]
				for thread in threads:
					thread.start()
				for thread in threads:
					thread.join()

			# Break EM update if converged
			updateDiff = rmse1d(F, F_prev)
			print "Inbreeding coefficients estimated (" + str(iteration) + "). RMSD=" + str(updateDiff)
			if updateDiff < EM_tole:
				print "EM (Inbreeding - sites) converged at iteration: " + str(iteration)
				break
			F_prev = np.copy(F)

	return F