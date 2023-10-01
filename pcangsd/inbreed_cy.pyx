# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
import numpy as np
cimport numpy as np
from cpython.mem cimport PyMem_RawMalloc, PyMem_RawFree
from cython.parallel import prange, parallel
from libc.math cimport log

##### Cython functions for inbreed.py #####
# Per-site
cpdef void inbreedSites_update(float[:,::1] L, float[:,::1] P, float[::1] F, int t) \
		nogil:
	cdef:
		int m = P.shape[0]
		int n = P.shape[1]
		int i, j
		float expH, obsH, p0, p1, p2, pSum, \
			tmp0, tmp1, tmp2, tmpSum, Fadj
	for j in prange(m, num_threads=t):
		expH = 0.0
		obsH = 0.0
		for i in range(n):
			Fadj = (1 - P[j,i])*P[j,i]*F[j]
			p0 = max(1e-4, (1 - P[j,i])*(1 - P[j,i]) + Fadj)
			p1 = max(1e-4, 2*P[j,i]*(1 - P[j,i]) - 2*Fadj)
			p2 = max(1e-4, P[j,i]*P[j,i] + Fadj)
			pSum = p0 + p1 + p2

			# Readjust distribution
			p0 = p0/pSum
			p1 = p1/pSum
			p2 = p2/pSum

			# Posterior
			tmp0 = L[j,2*i+0]*p0
			tmp1 = L[j,2*i+1]*p1
			tmp2 = (1.0 - L[j,2*i+0] - L[j,2*i+1])*p2
			tmpSum = tmp0 + tmp1 + tmp2

			# Sum over individuals
			obsH = obsH + tmp1/tmpSum

			# Count heterozygotes
			expH = expH + 2*P[j,i]*(1 - P[j,i])

		# ANGSD procedure
		obsH = max(1e-4, obsH/<float>(n))

		# Update the inbreeding coefficient
		F[j] = 1 - (n*obsH/expH)
		F[j] = min(max(-1.0, F[j]), 1.0)

# Per-site accelerated update
cpdef void inbreedSites_accel(float[:,::1] L, float[:,::1] P, float[::1] F, \
		float[::1] F_new, float[::1] d, int t) nogil:
	cdef:
		int m = P.shape[0]
		int n = P.shape[1]
		int i, j
		float expH, obsH, p0, p1, p2, pSum, \
			tmp0, tmp1, tmp2, tmpSum, Fadj
	for j in prange(m, num_threads=t):
		expH = 0.0
		obsH = 0.0
		for i in range(n):
			Fadj = (1 - P[j,i])*P[j,i]*F[j]
			p0 = max(1e-4, (1 - P[j,i])*(1 - P[j,i]) + Fadj)
			p1 = max(1e-4, 2*P[j,i]*(1 - P[j,i]) - 2*Fadj)
			p2 = max(1e-4, P[j,i]*P[j,i] + Fadj)
			pSum = p0 + p1 + p2

			# Readjust distribution
			p0 = p0/pSum
			p1 = p1/pSum
			p2 = p2/pSum

			# Posterior
			tmp0 = L[j,2*i+0]*p0
			tmp1 = L[j,2*i+1]*p1
			tmp2 = (1.0 - L[j,2*i+0] - L[j,2*i+1])*p2
			tmpSum = tmp0 + tmp1 + tmp2

			# Sum over individuals
			obsH = obsH + tmp1/tmpSum

			# Count heterozygotes
			expH = expH + 2*P[j,i]*(1 - P[j,i])

		# ANGSD procedure
		obsH = max(1e-4, obsH/<float>(n))

		# Update the inbreeding coefficient
		F_new[j] = 1 - (n*obsH/expH)
		F_new[j] = min(max(-1.0, F_new[j]), 1.0)
		d[j] = F_new[j] - F[j]

# Log-likelihoods
cpdef void loglike(float[:,::1] L, float[:,::1] P, float[::1] F, double[::1] T, \
		int t) nogil:
	cdef:
		int m = P.shape[0]
		int n = P.shape[1]
		int i, j
		float p0, p1, p2, pSum, Fadj
		double l0, l1, l2, logAlt, logNull
	for j in prange(m, num_threads=t):
		logAlt = 0.0
		logNull = 0.0
		for i in range(n):
			### Alternative model
			Fadj = (1 - P[j,i])*P[j,i]*F[j]
			p0 = max(1e-4, (1 - P[j,i])*(1 - P[j,i]) + Fadj)
			p1 = max(1e-4, 2*P[j,i]*(1 - P[j,i]) - 2*Fadj)
			p2 = max(1e-4, P[j,i]*P[j,i] + Fadj)
			pSum = p0 + p1 + p2

			# Readjust distribution
			p0 = p0/pSum
			p1 = p1/pSum
			p2 = p2/pSum

			# Likelihood*prior
			l0 = <double>(L[j,2*i+0]*p0)
			l1 = <double>(L[j,2*i+1]*p1)
			l2 = <double>((1.0 - L[j, 2*i+0] - L[j, 2*i+1])*p2)
			logAlt = logAlt + log(l0 + l1 + l2)

			### Null model
			l0 = <double>(L[j,2*i+0]*(1 - P[j,i])*(1 - P[j,i]))
			l1 = <double>(L[j,2*i+1]*2*P[j,i]*(1 - P[j,i]))
			l2 = <double>((1.0 - L[j,2*i+0] - L[j,2*i+1])*P[j,i]*P[j,i])
			logNull = logNull + log(l0 + l1 + l2)
		T[j] = 2.0*(logAlt - logNull)

# Per-individual
cpdef void inbreedSamples_update(float[:,::1] L, float[:,::1] P, float[::1] F, \
		float[::1] Ftmp, float[::1] Etmp, int t):
	cdef:
		int m = P.shape[0]
		int n = P.shape[1]
		int h, i, j, k, l
		float Fadj, p0, p1, p2, pSum, tmp0, tmp1, tmp2, tmpSum
		float* obsH
		float* expH
	for h in range(n):
		Ftmp[h] = 0.0
		Etmp[h] = 0.0
	with nogil, parallel(num_threads=t):
		obsH = <float*>PyMem_RawMalloc(sizeof(float)*n)
		expH = <float*>PyMem_RawMalloc(sizeof(float)*n)
		for l in range(n):
			obsH[l] = 0.0
			expH[l] = 0.0
		for j in prange(m):
			for i in range(n):
				Fadj = (1 - P[j,i])*P[j,i]*F[i]
				p0 = max(1e-4, (1 - P[j,i])*(1 - P[j,i]) + Fadj)
				p1 = max(1e-4, 2*P[j,i]*(1 - P[j,i]) - 2*Fadj)
				p2 = max(1e-4, P[j,i]*P[j,i] + Fadj)
				pSum = p0 + p1 + p2

				# Readjust distribution
				p0 = p0/pSum
				p1 = p1/pSum
				p2 = p2/pSum
				
				# Posterior
				tmp0 = L[j,2*i+0]*p0
				tmp1 = L[j,2*i+1]*p1
				tmp2 = (1.0 - L[j,2*i+0] - L[j,2*i+1])*p2
				tmpSum = tmp0 + tmp1 + tmp2

				# Sum over individuals
				obsH[i] += tmp1/tmpSum

				# Count heterozygotes
				expH[i] += 2*P[j,i]*(1 - P[j,i])
		with gil:
			for k in range(n):
				Ftmp[k] += obsH[k]
				Etmp[k] += expH[k]
		PyMem_RawFree(obsH)
		PyMem_RawFree(expH)
	for l in range(n):
		F[l] = 1.0 - Ftmp[l]/Etmp[l]

# Per-individual accelerated update
cpdef void inbreedSamples_accel(float[:,::1] L, float[:,::1] P, float[::1] F, \
		float[::1] F_new, float[::1] d, float[::1] Ftmp, float[::1] Etmp, int t):
	cdef:
		int m = P.shape[0]
		int n = P.shape[1]
		int h, i, j, k, l
		float Fadj, p0, p1, p2, pSum, tmp0, tmp1, tmp2, tmpSum
		float* obsH
		float* expH
	for h in range(n):
		Ftmp[h] = 0.0
		Etmp[h] = 0.0
	with nogil, parallel(num_threads=t):
		obsH = <float*>PyMem_RawMalloc(sizeof(float)*n)
		expH = <float*>PyMem_RawMalloc(sizeof(float)*n)
		for l in range(n):
			obsH[l] = 0.0
			expH[l] = 0.0
		for j in prange(m):
			for i in range(n):
				Fadj = (1 - P[j,i])*P[j,i]*F[i]
				p0 = max(1e-4, (1 - P[j,i])*(1 - P[j,i]) + Fadj)
				p1 = max(1e-4, 2*P[j,i]*(1 - P[j,i]) - 2*Fadj)
				p2 = max(1e-4, P[j,i]*P[j,i] + Fadj)
				pSum = p0 + p1 + p2

				# Readjust distribution
				p0 = p0/pSum
				p1 = p1/pSum
				p2 = p2/pSum
				
				# Posterior
				tmp0 = L[j,2*i+0]*p0
				tmp1 = L[j,2*i+1]*p1
				tmp2 = (1.0 - L[j,2*i+0] - L[j,2*i+1])*p2
				tmpSum = tmp0 + tmp1 + tmp2

				# Sum over individuals
				obsH[i] += tmp1/tmpSum

				# Count heterozygotes
				expH[i] += 2*P[j,i]*(1 - P[j,i])
		with gil:
			for k in range(n):
				Ftmp[k] += obsH[k]
				Etmp[k] += expH[k]
		PyMem_RawFree(obsH)
		PyMem_RawFree(expH)
	for l in range(n):
		F_new[l] = 1.0 - Ftmp[l]/Etmp[l]
		d[l] = F_new[l] - F[l]
