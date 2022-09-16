# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
import numpy as np
cimport numpy as np
from cython.parallel import prange, parallel
from libc.math cimport log
from libc.stdlib cimport malloc, free

##### Cython functions for inbreed.py #####
### Per-site
cpdef inbreedSites_update(float[:,::1] L, float[:,::1] P, float[::1] F, int t):
	cdef int m = P.shape[0]
	cdef int n = P.shape[1]
	cdef int i, s
	cdef float expH, obsH, prior0, prior1, prior2, priorSum, \
		temp0, temp1, temp2, tempSum, Fadj
	with nogil:
		for s in prange(m, num_threads=t):
			expH = 0.0
			obsH = 0.0
			for i in range(n):
				Fadj = (1 - P[s,i])*P[s,i]*F[s]
				prior0 = max(1e-4, (1 - P[s,i])*(1 - P[s,i]) + Fadj)
				prior1 = max(1e-4, 2*P[s,i]*(1 - P[s,i]) - 2*Fadj)
				prior2 = max(1e-4, P[s,i]*P[s,i] + Fadj)
				priorSum = prior0 + prior1 + prior2

				# Readjust distribution
				prior0 = prior0/priorSum
				prior1 = prior1/priorSum
				prior2 = prior2/priorSum

				# Posterior
				temp0 = L[s,2*i+0]*prior0
				temp1 = L[s,2*i+1]*prior1
				temp2 = (1.0 - L[s,2*i+0] - L[s,2*i+1])*prior2
				tempSum = temp0 + temp1 + temp2

				# Sum over individuals
				obsH = obsH + temp1/tempSum

				# Count heterozygotes
				expH = expH + 2*P[s,i]*(1 - P[s,i])

			# ANGSD procedure
			obsH = max(1e-4, obsH/<float>(n))

			# Update the inbreeding coefficient
			F[s] = 1 - (n*obsH/expH)
			F[s] = min(max(-1.0, F[s]), 1.0)

### Log-likelihoods
cpdef loglike(float[:,::1] L, float[:,::1] P, float[::1] F, \
		double[::1] T, int t):
	cdef int m = P.shape[0]
	cdef int n = P.shape[1]
	cdef int i, s
	cdef float prior0, prior1, prior2, priorSum, like0, like1, like2, Fadj
	cdef double logAlt, logNull
	with nogil:
		for s in prange(m, num_threads=t):
			logAlt = 0.0
			logNull = 0.0
			for i in range(n):
				### Alternative model
				Fadj = (1 - P[s,i])*P[s,i]*F[s]
				prior0 = max(1e-4, (1 - P[s,i])*(1 - P[s,i]) + Fadj)
				prior1 = max(1e-4, 2*P[s,i]*(1 - P[s,i]) - 2*Fadj)
				prior2 = max(1e-4, P[s,i]*P[s,i] + Fadj)
				priorSum = prior0 + prior1 + prior2

				# Readjust distribution
				prior0 = prior0/priorSum
				prior1 = prior1/priorSum
				prior2 = prior2/priorSum

				# Likelihood*prior
				like0 = L[s,2*i+0]*prior0
				like1 = L[s,2*i+1]*prior1
				like2 = (1.0 - L[s, 2*i+0] - L[s, 2*i+1])*prior2
				logAlt = logAlt + log(like0 + like1 + like2)

				### Null model
				like0 = L[s,2*i+0]*(1 - P[s,i])*(1 - P[s,i])
				like1 = L[s,2*i+1]*2*P[s,i]*(1 - P[s,i])
				like2 = (1.0 - L[s,2*i+0] - L[s,2*i+1])*P[s,i]*P[s,i]
				logNull = logNull + log(like0 + like1 + like2)
			T[s] = 2*(logAlt - logNull)

### Per-individual
cpdef inbreedSamples_update(float[:,::1] L, float[:,::1] P, float[::1] F, float[::1] Ftmp, \
		float[::1] Etmp, int t):
	cdef int m = P.shape[0]
	cdef int n = P.shape[1]
	cdef int i, j, k, l, s
	cdef float Fadj, prior0, prior1, prior2, priorSum, tmp0, tmp1, tmp2, tmpSum
	cdef float* obsH
	cdef float* expH
	for l in range(n):
		Ftmp[l] = 0.0
		Etmp[l] = 0.0
	with nogil, parallel(num_threads=t):
		obsH = <float*>malloc(sizeof(float)*n)
		expH = <float*>malloc(sizeof(float)*n)
		for j in range(n):
			obsH[j] = 0.0
			expH[j] = 0.0
		for s in prange(m):
			for i in range(n):
				Fadj = (1 - P[s,i])*P[s,i]*F[i]
				prior0 = max(1e-4, (1 - P[s,i])*(1 - P[s,i]) + Fadj)
				prior1 = max(1e-4, 2*P[s,i]*(1 - P[s,i]) - 2*Fadj)
				prior2 = max(1e-4, P[s,i]*P[s,i] + Fadj)
				priorSum = prior0 + prior1 + prior2

				# Readjust distribution
				prior0 = prior0/priorSum
				prior1 = prior1/priorSum
				prior2 = prior2/priorSum
				
				# Posterior
				tmp0 = L[s,2*i+0]*prior0
				tmp1 = L[s,2*i+1]*prior1
				tmp2 = (1.0 - L[s,2*i+0] - L[s,2*i+1])*prior2
				tmpSum = tmp0 + tmp1 + tmp2

				# Sum over individuals
				obsH[i] += tmp1/tmpSum

				# Count heterozygotes
				expH[i] += 2*P[s,i]*(1 - P[s,i])
		with gil:
			for k in range(n):
				Ftmp[k] += obsH[k]
				Etmp[k] += expH[k]
		free(obsH)
		free(expH)
	for l in range(n):
		F[l] = 1.0 - Ftmp[l]/Etmp[l]
