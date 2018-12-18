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
def inbreedEM_inner1(likeMatrix, Pi, S, N, F):
	m, n = Pi.shape # Dimensions

	for ind in xrange(S, min(S+N, m)):
		temp = 0

		# Estimate posterior probabilities (Z)
		for s in xrange(n):
			# Z = 0
			prob0 = likeMatrix[3*ind, s]*(1 - Pi[ind, s])*(1 - Pi[ind, s])
			prob1 = likeMatrix[3*ind+1, s]*2*Pi[ind, s]*(1 - Pi[ind, s])
			prob2 = likeMatrix[3*ind+2, s]*Pi[ind, s]*Pi[ind, s]
			Z0 = (prob0 + prob1 + prob2)*(1 - F[ind])

			# Z = 1
			prob0 = likeMatrix[3*ind, s]*(1 - Pi[ind, s])
			prob1 = 0
			prob2 = likeMatrix[3*ind+2, s]*Pi[ind, s]
			Z1 = (prob0 + prob1 + prob2)*F[ind]

			# Update the inbreeding coefficient
			temp += Z1/(Z0 + Z1)
		F[ind] = temp/n

# Inner update - model 2
@jit("void(f4[:, :], f4[:, :], i8, i8, f8[:])", nopython=True, nogil=True, cache=True)
def inbreedEM_inner2(likeMatrix, Pi, S, N, F):
	m, n = Pi.shape # Dimensions

	for ind in xrange(S, min(S+N, m)):
		expH = 0
		prob0 = 0
		prob1 = 0
		prob2 = 0

		for s in xrange(n):
			# Normalize priors
			Fadj = (1 - Pi[ind, s])*Pi[ind, s]*F[ind]
			prior0 = max(1e-4, (1 - Pi[ind, s])*(1 - Pi[ind, s]) + Fadj)
			prior1 = max(1e-4, 2*Pi[ind, s]*(1 - Pi[ind, s]) - 2*Fadj)
			prior2 = max(1e-4, Pi[ind, s]*Pi[ind, s] + Fadj)
			priorSum = prior0 + prior1 + prior2

			# Readjust genotype distribution
			prior0 /= priorSum
			prior1 /= priorSum
			prior2 /= priorSum

			# Estimate posterior probabilities
			temp0 = likeMatrix[3*ind, s]*prior0
			temp1 = likeMatrix[3*ind + 1, s]*prior1
			temp2 = likeMatrix[3*ind + 2, s]*prior2
			tempSum = temp0 + temp1 + temp2

			prob0 += temp0/tempSum
			prob1 += temp1/tempSum
			prob2 += temp2/tempSum

			# Counts of heterozygotes (expected)
			expH += 2*Pi[ind, s]*(1 - Pi[ind, s])
		
		# ANGSD procedure
		prob0 /= n
		prob1 /= n
		prob2 /= n
		prob0 = max(1e-4, prob0)
		prob1 = max(1e-4, prob1)
		prob2 = max(1e-4, prob2)
		prob1 /= (prob0 + prob1 + prob2)

		# Update the inbreeding coefficient
		F[ind] = 1 - (n*prob1/expH)
		F[ind] = max(-1.0, F[ind])
		F[ind] = min(1.0, F[ind])

# Population allele frequencies version (-iter 0) - Inner update - model 1
@jit("void(f4[:, :], f8[:], i8, i8, f8[:])", nopython=True, nogil=True, cache=True)
def inbreedEM_inner1_noPi(likeMatrix, f, S, N, F):
	m, n = likeMatrix.shape # Dimension of likelihood matrix
	m /= 3 # Number of individuals

	for ind in xrange(S, min(S+N, m)):
		temp = 0

		# Estimate posterior probabilities (Z)
		for s in xrange(n):
			# Z = 0
			prob0 = likeMatrix[3*ind, s]*(1 - f[s])*(1 - f[s])
			prob1 = likeMatrix[3*ind+1, s]*2*f[s]*(1 - f[s])
			prob2 = likeMatrix[3*ind+2, s]*f[s]*f[s]
			Z0 = (prob0 + prob1 + prob2)*(1 - F[ind])

			# Z = 1
			prob0 = likeMatrix[3*ind, s]*(1 - f[s])
			prob1 = 0
			prob2 = likeMatrix[3*ind+2, s]*f[s]
			Z1 = (prob0 + prob1 + prob2)*F[ind]

			# Update the inbreeding coefficient
			temp += Z1/(Z0 + Z1)
		F[ind] = temp/n

# Population allele frequencies version (-iter 0) - Inner update - model 2
@jit("void(f4[:, :], f8[:], i8, i8, f8[:])", nopython=True, nogil=True, cache=True)
def inbreedEM_inner2_noPi(likeMatrix, f, S, N, F):
	m, n = likeMatrix.shape # Dimension of likelihood matrix
	m /= 3 # Number of individuals
	expH = np.sum(2*f*(1 - f)) # Expected number of heterozygotes

	for ind in xrange(S, min(S+N, m)):
		prob0 = 0
		prob1 = 0
		prob2 = 0

		for s in xrange(n):
			# Normalize priors
			Fadj = (1 - f[s])*f[s]*F[ind]
			prior0 = max(1e-4, (1 - f[s])*(1 - f[s]) + Fadj)
			prior1 = max(1e-4, 2*f[s]*(1 - f[s]) - 2*Fadj)
			prior2 = max(1e-4, f[s]*f[s] + Fadj)
			priorSum = prior0 + prior1 + prior2

			# Readjust genotype distribution
			prior0 /= priorSum
			prior1 /= priorSum
			prior2 /= priorSum

			# Estimate posterior probabilities
			temp0 = likeMatrix[3*ind, s]*prior0
			temp1 = likeMatrix[3*ind + 1, s]*prior1
			temp2 = likeMatrix[3*ind + 2, s]*prior2
			tempSum = temp0 + temp1 + temp2

			prob0 += temp0/tempSum
			prob1 += temp1/tempSum
			prob2 += temp2/tempSum
		
		# ANGSD procedure
		prob0 /= n
		prob1 /= n
		prob2 /= n
		prob0 = max(1e-4, prob0)
		prob1 = max(1e-4, prob1)
		prob2 = max(1e-4, prob2)
		prob1 /= (prob0 + prob1 + prob2)

		# Update the inbreeding coefficient
		F[ind] = 1 - (n*prob1/expH)
		F[ind] = max(-1.0, F[ind])
		F[ind] = min(1.0, F[ind])


# EM algorithm for estimation of inbreeding coefficients
def inbreedEM(likeMatrix, Pi, model, EM=200, EM_tole=1e-4, t=1):
	m, n = likeMatrix.shape # Dimension of likelihood matrix
	m /= 3 # Number of individuals
	F = np.ones(m)*0.25 # Initialization of inbreeding coefficients
	F_prev = np.copy(F)

	# Multithreading parameters
	chunk_N = int(np.ceil(float(m)/t))
	chunks = [i * chunk_N for i in xrange(t)]

	# Model 1 - (Hall et al.)
	if model == 1:
		for iteration in xrange(1, EM + 1):
			if Pi.ndim == 2:
				# Multithreading - Update F
				threads = [threading.Thread(target=inbreedEM_inner1, args=(likeMatrix, Pi, chunk, chunk_N, F)) for chunk in chunks]
				for thread in threads:
					thread.start()
				for thread in threads:
					thread.join()
			else:
				# Population allele frequencies version (-iter 0) - Multithreading - Update F
				threads = [threading.Thread(target=inbreedEM_inner1_noPi, args=(likeMatrix, Pi, chunk, chunk_N, F)) for chunk in chunks]
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
			if Pi.ndim == 2:
				# Multithreading - Update F
				threads = [threading.Thread(target=inbreedEM_inner2, args=(likeMatrix, Pi, chunk, chunk_N, F)) for chunk in chunks]
				for thread in threads:
					thread.start()
				for thread in threads:
					thread.join()
			else:
				# Population allele frequencies version (-iter 0) - Multithreading - Update F
				threads = [threading.Thread(target=inbreedEM_inner2_noPi, args=(likeMatrix, Pi, chunk, chunk_N, F)) for chunk in chunks]
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