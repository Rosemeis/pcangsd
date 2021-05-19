import numpy as np
cimport numpy as np
from cython.parallel import prange
from cython import boundscheck, wraparound
from libc.math cimport sqrt

##### Cython functions for covariance.py ######
# Update posterior expectations (Fumagalli method)
@boundscheck(False)
@wraparound(False)
cpdef updateNormal(float[:,::1] L, float[::1] f, float[:,::1] E, int t):
	cdef int m = L.shape[0]
	cdef int n = L.shape[1]//2
	cdef int i, s
	cdef float p0, p1, p2
	with nogil:
		for s in prange(m, num_threads=t):
			for i in range(n):
				# Update dosage
				p0 = L[s,2*i+0]*(1 - f[s])*(1 - f[s])
				p1 = L[s,2*i+1]*2*f[s]*(1 - f[s])
				p2 = (1.0 - L[s, 2*i+0] - L[s, 2*i+1])*f[s]*f[s]
				E[s,i] = (p1 + 2*p2)/(p0 + p1 + p2) - 2*f[s]

# Update posterior expectations (PCAngsd method)
@boundscheck(False)
@wraparound(False)
cpdef updatePCAngsd(float[:,::1] L, float[::1] f, float[:,::1] P, float[:,::1] E, int t):
	cdef int m = L.shape[0]
	cdef int n = L.shape[1]//2
	cdef int i, s
	cdef float p0, p1, p2
	with nogil:
		for s in prange(m, num_threads=t):
			for i in range(n):
				# Update individual allele frequency
				P[s,i] = P[s,i] + 2*f[s]
				P[s,i] = P[s,i]/2
				P[s,i] = min(max(P[s,i], 1e-4), 1-(1e-4))

				# Center dosage
				p0 = L[s,2*i+0]*(1 - P[s,i])*(1 - P[s,i])
				p1 = L[s,2*i+1]*2*P[s,i]*(1 - P[s,i])
				p2 = (1.0 - L[s, 2*i+0] - L[s, 2*i+1])*P[s,i]*P[s,i]
				E[s,i] = (p1 + 2*p2)/(p0 + p1 + p2) - 2*f[s]

# Standardize posterior expectations (Fumagalli method)
@boundscheck(False)
@wraparound(False)
cpdef covNormal(float[:,::1] L, float[::1] f, float[:,::1] E, float[::1] dCov, \
				int t):
	cdef int m = L.shape[0]
	cdef int n = L.shape[1]//2
	cdef int i, s
	cdef float p0, p1, p2, pSum, temp
	with nogil:
		for s in prange(m, num_threads=t):
			for i in range(n):
				# Standardize dosage
				p0 = L[s,2*i+0]*(1 - f[s])*(1 - f[s])
				p1 = L[s,2*i+1]*2*f[s]*(1 - f[s])
				p2 = (1.0 - L[s, 2*i+0] - L[s, 2*i+1])*f[s]*f[s]
				pSum = p0 + p1 + p2
				E[s,i] = (p1 + 2*p2)/pSum - 2*f[s]
				E[s,i] = E[s,i]/sqrt(2*f[s]*(1 - f[s]))

				# Estimate diagonal
				temp = (0 - 2*f[s])*(0 - 2*f[s])*(p0/pSum)
				temp = temp + (1 - 2*f[s])*(1 - 2*f[s])*(p1/pSum)
				temp = temp + (2 - 2*f[s])*(2 - 2*f[s])*(p2/pSum)
				dCov[i] = dCov[i] + temp/(2*f[s]*(1 - f[s]))

# Standardize posterior expectations (PCAngsd method)
@boundscheck(False)
@wraparound(False)
cpdef covPCAngsd(float[:,::1] L, float[::1] f, float[:,::1] P, float[:,::1] E, \
					float[::1] dCov, int t):
	cdef int m = L.shape[0]
	cdef int n = L.shape[1]//2
	cdef int i, s
	cdef float p0, p1, p2, pSum, temp
	with nogil:
		for s in prange(m, num_threads=t):
			for i in range(n):
				# Update individual allele frequency
				P[s,i] = P[s,i] + 2*f[s]
				P[s,i] = P[s,i]/2
				P[s,i] = min(max(P[s,i], 1e-4), 1-(1e-4))

				# Standardize dosage
				p0 = L[s,2*i+0]*(1 - P[s,i])*(1 - P[s,i])
				p1 = L[s,2*i+1]*2*P[s,i]*(1 - P[s,i])
				p2 = (1.0 - L[s, 2*i+0] - L[s, 2*i+1])*P[s,i]*P[s,i]
				pSum = p0 + p1 + p2
				E[s,i] = (p1 + 2*p2)/pSum - 2*f[s]
				E[s,i] = E[s,i]/sqrt(2*f[s]*(1 - f[s]))

				# Estimate diagonal
				temp = (0 - 2*f[s])*(0 - 2*f[s])*(p0/pSum)
				temp = temp + (1 - 2*f[s])*(1 - 2*f[s])*(p1/pSum)
				temp = temp + (2 - 2*f[s])*(2 - 2*f[s])*(p2/pSum)
				dCov[i] = dCov[i] + temp/(2*f[s]*(1 - f[s]))

# Standardize posterior expectations for selection
@boundscheck(False)
@wraparound(False)
cpdef updateSelection(float[:,::1] L, float[::1] f, float[:,::1] P, \
						float[:,::1] E, int t):
	cdef int m = L.shape[0]
	cdef int n = L.shape[1]//2
	cdef int i, s
	cdef float p0, p1, p2
	with nogil:
		for s in prange(m, num_threads=t):
			for i in range(n):
				# Standardize dosage
				p0 = L[s,2*i+0]*(1 - P[s,i])*(1 - P[s,i])
				p1 = L[s,2*i+1]*2*P[s,i]*(1 - P[s,i])
				p2 = (1.0 - L[s, 2*i+0] - L[s, 2*i+1])*P[s,i]*P[s,i]
				E[s,i] = (p1 + 2*p2)/(p0 + p1 + p2) - 2*f[s]
				E[s,i] = E[s,i]/sqrt(2*f[s]*(1 - f[s]))

# Update dosages for saving
@boundscheck(False)
@wraparound(False)
cpdef updateDosages(float[:,::1] L, float[:,::1] P, float[:,::1] E, int t):
	cdef int m = L.shape[0]
	cdef int n = L.shape[1]//2
	cdef int i, s
	cdef float p0, p1, p2
	with nogil:
		for s in prange(m, num_threads=t):
			for i in range(n):
				# Update dosage
				p0 = L[s,2*i+0]*(1 - P[s,i])*(1 - P[s,i])
				p1 = L[s,2*i+1]*2*P[s,i]*(1 - P[s,i])
				p2 = (1.0 - L[s, 2*i+0] - L[s, 2*i+1])*P[s,i]*P[s,i]
				E[s,i] = (p1 + 2*p2)/(p0 + p1 + p2)
