import numpy as np
cimport numpy as np
from cython.parallel import prange
from cython import boundscheck, wraparound
from libc.math cimport log

##### Cython functions for EM inbreeding functions ######

### Per-site
@boundscheck(False)
@wraparound(False)
cpdef emInbreedSites_update(float[:,::1] L, float[:,::1] Pi, float[::1] F, int t):
	cdef int n = Pi.shape[0]
	cdef int m = Pi.shape[1]
	cdef int i, j
	cdef float expH, prob0, prob1, prob2, prior0, prior1, prior2, priorSum, temp0, temp1, temp2, tempSum, Fadj
	with nogil:
		for j in prange(m, num_threads=t):
			expH = 0.0
			prob0 = 0.0
			prob1 = 0.0
			prob2 = 0.0

			for i in range(n):
				Fadj = (1 - Pi[i,j])*Pi[i,j]*F[j]
				prior0 = max(1e-4, (1 - Pi[i,j])*(1 - Pi[i,j]) + Fadj)
				prior1 = max(1e-4, 2*Pi[i,j]*(1 - Pi[i,j]) - 2*Fadj)
				prior2 = max(1e-4, Pi[i,j]*Pi[i,j] + Fadj)
				priorSum = prior0 + prior1 + prior2

				# Readjust genotype distribution
				prior0 = prior0/priorSum
				prior1 = prior1/priorSum
				prior2 = prior2/priorSum

				# Estimate posterior probabilities
				temp0 = L[3*i,j]*prior0
				temp1 = L[3*i+1,j]*prior1
				temp2 = L[3*i+2,j]*prior2
				tempSum = temp0 + temp1 + temp2

				prob0 = prob0 + temp0/tempSum
				prob1 = prob1 + temp1/tempSum
				prob2 = prob2 + temp2/tempSum

				# Counts of heterozygotes (expected)
				expH = expH + 2*Pi[i,j]*(1 - Pi[i,j])

			# ANGSD procedure
			prob0 = max(1e-4, prob0/n)
			prob1 = max(1e-4, prob1/n)
			prob2 = max(1e-4, prob2/n)
			prob1 = prob1/(prob0 + prob1 + prob2)

			# Update the inbreeding coefficient
			F[j] = 1 - (n*prob1/expH)
			F[j] = min(max(-1.0, F[j]), 1.0)

# Estimate log-likelihoods
@boundscheck(False)
@wraparound(False)
cpdef loglike(float[:,::1] L, float[:,::1] Pi, float[::1] F, float[::1] lrt, int t):
	cdef int n = Pi.shape[0]
	cdef int m = Pi.shape[1]
	cdef int i, j
	cdef float prior0, prior1, prior2, priorSum, like0, like1, like2, Fadj, logAlt, logNull
	with nogil:
		for j in prange(m, num_threads=t):
			logAlt = 0.0
			logNull = 0.0
			for i in range(n):
				### Alternative model
				# Priors
				Fadj = (1 - Pi[i,j])*Pi[i,j]*F[j]
				prior0 = max(1e-4, (1 - Pi[i,j])*(1 - Pi[i,j]) + Fadj)
				prior1 = max(1e-4, 2*Pi[i,j]*(1 - Pi[i,j]) - 2*Fadj)
				prior2 = max(1e-4, Pi[i,j]*Pi[i,j] + Fadj)
				priorSum = prior0 + prior1 + prior2

				# Readjust genotype distribution
				prior0 = prior0/priorSum
				prior1 = prior1/priorSum
				prior2 = prior2/priorSum

				# Likelihood*prior
				like0 = L[3*i,j]*prior0
				like1 = L[3*i+1,j]*prior1
				like2 = L[3*i+2,j]*prior2
				logAlt = logAlt + log(like0 + like1 + like2)

				### Null model
				like0 = L[3*i,j]*(1 - Pi[i,j])*(1 - Pi[i,j])
				like1 = L[3*i+1,j]*2*Pi[i,j]*(1 - Pi[i,j])
				like2 = L[3*i+2,j]*Pi[i,j]*Pi[i,j]
				logNull = logNull + log(like0 + like1 + like2)
			lrt[j] = 2*(logAlt - logNull)

### Per-individual
# Simple
@boundscheck(False)
@wraparound(False)
cpdef emInbreed_update(float[:,::1] L, float[:,::1] Pi, float[::1] F, int t):
	cdef int n = Pi.shape[0]
	cdef int m = Pi.shape[1]
	cdef int i, j
	cdef float expH, prob0, prob1, prob2, prior0, prior1, prior2, priorSum, temp0, temp1, temp2, tempSum, Fadj
	with nogil:
		for i in prange(n, num_threads=t):
			expH = 0.0
			prob0 = 0.0
			prob1 = 0.0
			prob2 = 0.0
			
			for j in range(m):
				Fadj = (1 - Pi[i,j])*Pi[i,j]*F[i]
				prior0 = max(1e-4, (1 - Pi[i,j])*(1 - Pi[i,j]) + Fadj)
				prior1 = max(1e-4, 2*Pi[i,j]*(1 - Pi[i,j]) - 2*Fadj)
				prior2 = max(1e-4, Pi[i,j]*Pi[i,j] + Fadj)
				priorSum = prior0 + prior1 + prior2

				# Readjust genotype distribution
				prior0 = prior0/priorSum
				prior1 = prior1/priorSum
				prior2 = prior2/priorSum

				# Estimate posterior probabilities
				temp0 = L[3*i,j]*prior0
				temp1 = L[3*i+1,j]*prior1
				temp2 = L[3*i+2,j]*prior2
				tempSum = temp0 + temp1 + temp2

				prob0 = prob0 + temp0/tempSum
				prob1 = prob1 + temp1/tempSum
				prob2 = prob2 + temp2/tempSum

				# Counts of heterozygotes (expected)
				expH = expH + 2*Pi[i,j]*(1 - Pi[i,j])
			
			# ANGSD procedure
			prob0 = max(1e-4, prob0/m)
			prob1 = max(1e-4, prob1/m)
			prob2 = max(1e-4, prob2/m)
			prob1 = prob1/(prob0 + prob1 + prob2)

			# Update the inbreeding coefficient
			F[i] = 1 - (m*prob1/expH)
			F[i] = min(max(-1.0, F[i]), 1.0)

# Hall
@boundscheck(False)
@wraparound(False)
cpdef emHall_update(float[:,::1] L, float[:,::1] Pi, float[::1] F, int t):
	cdef int n = Pi.shape[0]
	cdef int m = Pi.shape[1]
	cdef int i, j
	cdef float temp, prob0, prob1, prob2, Z0, Z1
	with nogil:
		for i in prange(n, num_threads=t):
			temp = 0.0

			for j in range(m):
				# Z = 0
				prob0 = L[3*i,j]*(1 - Pi[i,j])*(1 - Pi[i,j])
				prob1 = L[3*i+1,j]*2*Pi[i,j]*(1 - Pi[i,j])
				prob2 = L[3*i+2,j]*Pi[i,j]*Pi[i,j]
				Z0 = (prob0 + prob1 + prob2)*(1 - F[i])

				# Z = 1
				prob0 = L[3*i,j]*(1 - Pi[i,j])
				prob1 = 0.0
				prob2 = L[3*i+2,j]*Pi[i,j]
				Z1 = (prob0 + prob1 + prob2)*F[i]

				# Update the inbreeding coefficient
				temp = temp + Z1/(Z0 + Z1)
			F[i] = temp/m