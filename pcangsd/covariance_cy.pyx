# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
import numpy as np
cimport numpy as np
from cpython.mem cimport PyMem_RawCalloc, PyMem_RawFree
from cython.parallel import prange, parallel
from libc.math cimport sqrt

##### Cython functions for covariance.py ######
# Update posterior expectations (Fumagalli method)
cpdef void updateNormal(float[:,::1] L, float[:,::1] E, double[::1] f, int t) \
		noexcept nogil:
	cdef:
		int m = L.shape[0]
		int n = L.shape[1]//2
		int i, j
		double f1, p0, p1, p2
	for j in prange(m, num_threads=t):
		f1 = <double>f[j]
		for i in range(n):
			# Update dosage
			p0 = L[j,2*i+0]*(1.0-f1)*(1.0-f1)
			p1 = L[j,2*i+1]*2.0*f1*(1.0-f1)
			p2 = (1.0 - L[j,2*i+0] - L[j,2*i+1])*f1*f1
			E[j,i] = (p1 + 2.0*p2)/(p0 + p1 + p2) - 2.0*f1

# Update posterior expectations (PCAngsd method)
cpdef void updatePCAngsd(float[:,::1] L, float[:,::1] P, \
		float[:,::1] E, double[::1] f, int t) noexcept nogil:
	cdef:
		int m = L.shape[0]
		int n = L.shape[1]//2
		int i, j
		double p0, p1, p2
	for j in prange(m, num_threads=t):
		for i in range(n):
			# Update individual allele frequency
			P[j,i] = P[j,i] + 2.0*f[j]
			P[j,i] = P[j,i]/2.0
			P[j,i] = min(max(P[j,i], 1e-4), 1-(1e-4))

			# Center dosage
			p0 = L[j,2*i+0]*(1.0-P[j,i])*(1.0-P[j,i])
			p1 = L[j,2*i+1]*2.0*P[j,i]*(1.0-P[j,i])
			p2 = (1.0 - L[j,2*i+0] - L[j,2*i+1])*P[j,i]*P[j,i]
			E[j,i] = (p1 + 2.0*p2)/(p0 + p1 + p2) - 2.0*f[j]

# Standardize posterior expectations (Fumagalli method)
cpdef void covNormal(float[:,::1] L, float[:,::1] E, double[::1] f, double[::1] dCov, \
		int t) noexcept nogil:
	cdef:
		int m = L.shape[0]
		int n = L.shape[1]//2
		int i, j, k
		float f1, f2, p0, p1, p2, pSum, tmp
		double* dPrivate
	with nogil, parallel(num_threads=t):
		dPrivate = <double*>PyMem_RawCalloc(n, sizeof(double))
		for j in prange(m):
			f1 = f[j]
			f2 = 2.0*f1
			for i in range(n):
				# Standardize dosage
				p0 = L[j,2*i+0]*(1.0-f1)*(1.0-f1)
				p1 = L[j,2*i+1]*f2*(1.0-f1)
				p2 = (1.0 - L[j,2*i+0] - L[j,2*i+1])*f1*f1
				pSum = p0 + p1 + p2
				E[j,i] = (p1 + 2.0*p2)/pSum - f2
				E[j,i] = E[j,i]/sqrt(f2*(1.0-f1))

				# Estimate diagonal
				tmp = (0.0-f2)*(0.0-f2)*(p0/pSum)
				tmp = tmp + (1.0-f2)*(1.0-f2)*(p1/pSum)
				tmp = tmp + (2.0-f2)*(2.0-f2)*(p2/pSum)
				dPrivate[i] = dPrivate[i] + tmp/(f2*(1.0-f1))
		with gil:
			for k in range(n):
				dCov[k] += dPrivate[k]
		PyMem_RawFree(dPrivate)

# Standardize posterior expectations (PCAngsd method)
cpdef void covPCAngsd(float[:,::1] L, float[:,::1] P, float[:,::1] E, \
		double[::1] f, double[::1] dCov, int t) noexcept nogil:
	cdef:
		int m = L.shape[0]
		int n = L.shape[1]//2
		int i, j, k
		float f1, f2, p0, p1, p2, pSum, tmp
		double* dPrivate
	with nogil, parallel(num_threads=t):
		dPrivate = <double*>PyMem_RawCalloc(n, sizeof(double))
		for j in prange(m):
			f1 = f[j]
			f2 = 2.0*f1
			for i in range(n):
				# Update individual allele frequency
				P[j,i] = P[j,i] + f2
				P[j,i] = P[j,i]/2
				P[j,i] = min(max(P[j,i], 1e-4), 1-(1e-4))

				# Standardize dosage
				p0 = L[j,2*i+0]*(1.0-P[j,i])*(1.0-P[j,i])
				p1 = L[j,2*i+1]*2.0*P[j,i]*(1.0-P[j,i])
				p2 = (1.0 - L[j,2*i+0] - L[j,2*i+1])*P[j,i]*P[j,i]
				pSum = p0 + p1 + p2
				E[j,i] = (p1 + 2.0*p2)/pSum - f2
				E[j,i] = E[j,i]/sqrt(f2*(1.0-f1))

				# Estimate diagonal
				tmp = (0.0-f2)*(0.0-f2)*(p0/pSum)
				tmp = tmp + (1.0-f2)*(1.0-f2)*(p1/pSum)
				tmp = tmp + (2.0-f2)*(2.0-f2)*(p2/pSum)
				dPrivate[i] = dPrivate[i] + tmp/(f2*(1.0-f1))
		with gil:
			for k in range(n):
				dCov[k] += dPrivate[k]
		PyMem_RawFree(dPrivate)

# Standardize posterior expectations for selection
cpdef void updateSelection(float[:,::1] L, float[:,::1] P, float[:,::1] E, \
		double[::1] f, int t) noexcept nogil:
	cdef:
		int m = L.shape[0]
		int n = L.shape[1]//2
		int i, j
		double p0, p1, p2
	for j in prange(m, num_threads=t):
		for i in range(n):
			# Standardize dosage
			p0 = L[j,2*i+0]*(1.0-P[j,i])*(1.0-P[j,i])
			p1 = L[j,2*i+1]*2.0*P[j,i]*(1.0-P[j,i])
			p2 = (1.0 - L[j,2*i+0] - L[j,2*i+1])*P[j,i]*P[j,i]
			E[j,i] = (p1 + 2.0*p2)/(p0 + p1 + p2) - 2.0*f[j]
			E[j,i] = E[j,i]/sqrt(2.0*f[j]*(1.0-f[j]))

# Update dosages for saving
cpdef void updateDosages(float[:,::1] L, float[:,::1] P, float[:,::1] E, int t) \
		noexcept nogil:
	cdef:
		int m = L.shape[0]
		int n = L.shape[1]//2
		int i, j
		double p0, p1, p2
	for j in prange(m, num_threads=t):
		for i in range(n):
			# Update dosage
			p0 = L[j,2*i+0]*(1.0-P[j,i])*(1.0-P[j,i])
			p1 = L[j,2*i+1]*2.0*P[j,i]*(1.0-P[j,i])
			p2 = (1.0 - L[j,2*i+0] - L[j,2*i+1])*P[j,i]*P[j,i]
			E[j,i] = (p1 + 2.0*p2)/(p0 + p1 + p2)
