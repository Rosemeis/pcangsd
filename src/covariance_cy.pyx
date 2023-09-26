# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
import numpy as np
cimport numpy as np
from cpython.mem cimport PyMem_RawMalloc, PyMem_RawFree
from cython.parallel import prange, parallel
from libc.math cimport sqrt

##### Cython functions for covariance.py ######
# Update posterior expectations (Fumagalli method)
cpdef void updateNormal(float[:,::1] L, float[::1] f, float[:,::1] E, int t) nogil:
	cdef:
		int m = L.shape[0]
		int n = L.shape[1]//2
		int i, j
		float p0, p1, p2
	for j in prange(m, num_threads=t):
		for i in range(n):
			# Update dosage
			p0 = L[j,2*i+0]*(1 - f[j])*(1 - f[j])
			p1 = L[j,2*i+1]*2*f[j]*(1 - f[j])
			p2 = (1.0 - L[j,2*i+0] - L[j,2*i+1])*f[j]*f[j]
			E[j,i] = (p1 + 2*p2)/(p0 + p1 + p2) - 2*f[j]

# Update posterior expectations (PCAngsd method)
cpdef void updatePCAngsd(float[:,::1] L, float[::1] f, float[:,::1] P, \
		float[:,::1] E, int t) nogil:
	cdef:
		int m = L.shape[0]
		int n = L.shape[1]//2
		int i, j
		float p0, p1, p2
	for j in prange(m, num_threads=t):
		for i in range(n):
			# Update individual allele frequency
			P[j,i] = P[j,i] + 2*f[j]
			P[j,i] = P[j,i]/2
			P[j,i] = min(max(P[j,i], 1e-4), 1-(1e-4))

			# Center dosage
			p0 = L[j,2*i+0]*(1 - P[j,i])*(1 - P[j,i])
			p1 = L[j,2*i+1]*2*P[j,i]*(1 - P[j,i])
			p2 = (1.0 - L[j,2*i+0] - L[j,2*i+1])*P[j,i]*P[j,i]
			E[j,i] = (p1 + 2*p2)/(p0 + p1 + p2) - 2*f[j]

# Standardize posterior expectations (Fumagalli method)
cpdef void covNormal(float[:,::1] L, float[::1] f, float[:,::1] E, float[::1] dCov, \
		int t):
	cdef:
		int m = L.shape[0]
		int n = L.shape[1]//2
		int i, j, k, l
		float p0, p1, p2, pSum, tmp
		float* dPrivate
	with nogil, parallel(num_threads=t):
		dPrivate = <float*>PyMem_RawMalloc(sizeof(float)*n)
		for l in range(n):
			dPrivate[l] = 0.0
		for j in prange(m):
			for i in range(n):
				# Standardize dosage
				p0 = L[j,2*i+0]*(1 - f[j])*(1 - f[j])
				p1 = L[j,2*i+1]*2*f[j]*(1 - f[j])
				p2 = (1.0 - L[j,2*i+0] - L[j,2*i+1])*f[j]*f[j]
				pSum = p0 + p1 + p2
				E[j,i] = (p1 + 2*p2)/pSum - 2*f[j]
				E[j,i] = E[j,i]/sqrt(2*f[j]*(1 - f[j]))

				# Estimate diagonal
				tmp = (0 - 2*f[j])*(0 - 2*f[j])*(p0/pSum)
				tmp = tmp + (1 - 2*f[j])*(1 - 2*f[j])*(p1/pSum)
				tmp = tmp + (2 - 2*f[j])*(2 - 2*f[j])*(p2/pSum)
				dPrivate[i] += tmp/(2*f[j]*(1 - f[j]))
		with gil:
			for k in range(n):
				dCov[k] += dPrivate[k]
		PyMem_RawFree(dPrivate)

# Standardize posterior expectations (PCAngsd method)
cpdef void covPCAngsd(float[:,::1] L, float[::1] f, float[:,::1] P, float[:,::1] E, \
		float[::1] dCov, int t):
	cdef:
		int m = L.shape[0]
		int n = L.shape[1]//2
		int i, j, k, l
		float p0, p1, p2, pSum, tmp
		float* dPrivate
	with nogil, parallel(num_threads=t):
		dPrivate = <float*>PyMem_RawMalloc(sizeof(float)*n)
		for l in range(n):
			dPrivate[l] = 0.0
		for j in prange(m):
			for i in range(n):
				# Update individual allele frequency
				P[j,i] = P[j,i] + 2*f[j]
				P[j,i] = P[j,i]/2
				P[j,i] = min(max(P[j,i], 1e-4), 1-(1e-4))

				# Standardize dosage
				p0 = L[j,2*i+0]*(1 - P[j,i])*(1 - P[j,i])
				p1 = L[j,2*i+1]*2*P[j,i]*(1 - P[j,i])
				p2 = (1.0 - L[j,2*i+0] - L[j,2*i+1])*P[j,i]*P[j,i]
				pSum = p0 + p1 + p2
				E[j,i] = (p1 + 2*p2)/pSum - 2*f[j]
				E[j,i] = E[j,i]/sqrt(2*f[j]*(1 - f[j]))

				# Estimate diagonal
				tmp = (0 - 2*f[j])*(0 - 2*f[j])*(p0/pSum)
				tmp = tmp + (1 - 2*f[j])*(1 - 2*f[j])*(p1/pSum)
				tmp = tmp + (2 - 2*f[j])*(2 - 2*f[j])*(p2/pSum)
				dPrivate[i] += tmp/(2*f[j]*(1 - f[j]))
		with gil:
			for k in range(n):
				dCov[k] += dPrivate[k]
		PyMem_RawFree(dPrivate)

# Standardize posterior expectations for selection
cpdef void updateSelection(float[:,::1] L, float[::1] f, float[:,::1] P, \
		float[:,::1] E, int t) nogil:
	cdef:
		int m = L.shape[0]
		int n = L.shape[1]//2
		int i, j
		float p0, p1, p2
	for j in prange(m, num_threads=t):
		for i in range(n):
			# Standardize dosage
			p0 = L[j,2*i+0]*(1 - P[j,i])*(1 - P[j,i])
			p1 = L[j,2*i+1]*2*P[j,i]*(1 - P[j,i])
			p2 = (1.0 - L[j,2*i+0] - L[j,2*i+1])*P[j,i]*P[j,i]
			E[j,i] = (p1 + 2*p2)/(p0 + p1 + p2) - 2*f[j]
			E[j,i] = E[j,i]/sqrt(2*f[j]*(1 - f[j]))

# Update dosages for saving
cpdef void updateDosages(float[:,::1] L, float[:,::1] P, float[:,::1] E, int t) nogil:
	cdef:
		int m = L.shape[0]
		int n = L.shape[1]//2
		int i, j
		float p0, p1, p2
	for j in prange(m, num_threads=t):
		for i in range(n):
			# Update dosage
			p0 = L[j,2*i+0]*(1 - P[j,i])*(1 - P[j,i])
			p1 = L[j,2*i+1]*2*P[j,i]*(1 - P[j,i])
			p2 = (1.0 - L[j,2*i+0] - L[j,2*i+1])*P[j,i]*P[j,i]
			E[j,i] = (p1 + 2*p2)/(p0 + p1 + p2)
