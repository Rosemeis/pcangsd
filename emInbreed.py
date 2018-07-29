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
	probMatrix = np.empty((3, n)) # Container for posterior probabilities

	for ind in xrange(S, min(S+N, m)):
		expH = 0

		# Estimate posterior probabilities
		for s in xrange(n):
			probMatrix[0, s] = max(0, likeMatrix[3*ind, s]*((1 - indF[ind, s])*(1 - indF[ind, s]) + (1 - indF[ind, s])*indF[ind, s]*F[ind]))
			probMatrix[1, s] = max(0, likeMatrix[3*ind + 1, s]*2*indF[ind, s]*(1 - indF[ind, s])*(1 - F[ind]))
			probMatrix[2, s] = max(0, likeMatrix[3*ind + 2, s]*(indF[ind, s]*indF[ind, s] + (1 - indF[ind, s])*indF[ind, s]*F[ind]))
			sumNorm = np.sum(probMatrix[:, s])

			# Normalize posteriors
			if sumNorm > 0:
				probMatrix[1, s] /= sumNorm
			probMatrix[1, s] = max(1e-4, probMatrix[1, s]) # Fix lower boundary

			# Expected number of heterozygotes
			expH += 2*indF[ind, s]*(1 - indF[ind, s])
		
		# Update the inbreeding coefficient
		F[ind] = 1 - (np.sum(probMatrix[1, :])/expH)

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
	probMatrix = np.empty((3, n)) # Container for posterior probabilities
	expH = np.sum(2*f*(1 - f)) # Expected number of heterozygotes

	for ind in xrange(S, min(S+N, m)):
		# Estimate posterior probabilities
		for s in xrange(n):
			probMatrix[0, s] = max(0, likeMatrix[3*ind, s]*((1 - f[s])*(1 - f[s]) + (1 - f[s])*f[s]*F[ind]))
			probMatrix[1, s] = max(0, likeMatrix[3*ind + 1, s]*2*f[s]*(1 - f[s])*(1 - F[ind]))
			probMatrix[2, s] = max(0, likeMatrix[3*ind + 2, s]*(f[s]*f[s] + (1 - f[s])*f[s]*F[ind]))
			sumNorm = np.sum(probMatrix[:, s])

			# Normalize posteriors
			if sumNorm > 0:
				probMatrix[1, s] /= sumNorm
			probMatrix[1, s] = max(1e-4, probMatrix[1, s]) # Fix lower boundary
		
		# Update the inbreeding coefficient
		F[ind] = 1 - (np.sum(probMatrix[1, :])/expH)


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