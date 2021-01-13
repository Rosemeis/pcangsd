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
	cdef int n = L.shape[1]//3
	cdef int i, s
	cdef float p0, p1, p2
	with nogil:
		for s in prange(m, num_threads=t):
			for i in range(n):
				p0 = L[s,3*i+0]*(1 - f[s])*(1 - f[s])
				p1 = L[s,3*i+1]*2*f[s]*(1 - f[s])
				p2 = L[s,3*i+2]*f[s]*f[s]

				# Update dosage
				E[s,i] = (p1 + 2*p2)/(p0 + p1 + p2)

# Update posterior expectations (PCAngsd method)
@boundscheck(False)
@wraparound(False)
cpdef updatePCAngsd(float[:,::1] L, float[:,::1] P, float[:,::1] E, int t):
	cdef int m = L.shape[0]
	cdef int n = L.shape[1]//3
	cdef int i, s
	cdef float p0, p1, p2
	with nogil:
		for s in prange(m, num_threads=t):
			for i in range(n):
				p0 = L[s,3*i+0]*(1 - P[s,i])*(1 - P[s,i])
				p1 = L[s,3*i+1]*2*P[s,i]*(1 - P[s,i])
				p2 = L[s,3*i+2]*P[s,i]*P[s,i]

				# Update dosage
				E[s,i] = (p1 + 2*p2)/(p0 + p1 + p2)

# Update posterior expectations including cov diagonal (Fumagalli method)
@boundscheck(False)
@wraparound(False)
cpdef covNormal(float[:,::1] L, float[::1] f, float[:,::1] E, float[::1] dCov, int t):
	cdef int m = L.shape[0]
	cdef int n = L.shape[1]//3
	cdef int i, s
	cdef float p0, p1, p2, pSum, temp
	with nogil:
		for s in prange(m, num_threads=t):
			for i in range(n):
				p0 = L[s,3*i+0]*(1 - f[s])*(1 - f[s])
				p1 = L[s,3*i+1]*2*f[s]*(1 - f[s])
				p2 = L[s,3*i+2]*f[s]*f[s]
				pSum = p0 + p1 + p2

				# Update dosage and cov diagonal
				E[s,i] = (p1 + 2*p2)/pSum
				temp = (0 - 2*f[s])*(0 - 2*f[s])*(p0/pSum)
				temp = temp + (1 - 2*f[s])*(1 - 2*f[s])*(p1/pSum)
				temp = temp + (2 - 2*f[s])*(2 - 2*f[s])*(p2/pSum)
				dCov[i] = dCov[i] + temp/(2*f[s]*(1 - f[s]))

# Update posterior expectations including cov diagonal (PCAngsd method)
@boundscheck(False)
@wraparound(False)
cpdef covPCAngsd(float[:,::1] L, float[::1] f, float[:,::1] P, float[:,::1] E, float[::1] dCov, int t):
	cdef int m = L.shape[0]
	cdef int n = L.shape[1]//3
	cdef int i, s
	cdef float p0, p1, p2, pSum, temp
	with nogil:
		for s in prange(m, num_threads=t):
			for i in range(n):
				p0 = L[s,3*i+0]*(1 - P[s,i])*(1 - P[s,i])
				p1 = L[s,3*i+1]*2*P[s,i]*(1 - P[s,i])
				p2 = L[s,3*i+2]*P[s,i]*P[s,i]
				pSum = p0 + p1 + p2

				# Update dosage and cov diagonal
				E[s,i] = (p1 + 2*p2)/pSum
				temp = (0 - 2*f[s])*(0 - 2*f[s])*(p0/pSum)
				temp = temp + (1 - 2*f[s])*(1 - 2*f[s])*(p1/pSum)
				temp = temp + (2 - 2*f[s])*(2 - 2*f[s])*(p2/pSum)
				dCov[i] = dCov[i] + temp/(2*f[s]*(1 - f[s]))

# Center posterior expectations
@boundscheck(False)
@wraparound(False)
cpdef centerE(float[:,::1] E, float[::1] f, int t):
	cdef int m = E.shape[0]
	cdef int n = E.shape[1]
	cdef int i, s
	with nogil:
		for s in prange(m, num_threads=t):
			for i in range(n):
				E[s,i] = E[s,i] - 2*f[s]

# Standardize posterior expectations
@boundscheck(False)
@wraparound(False)
cpdef standardizeE(float[:,::1] E, float[::1] f, int t):
	cdef int m = E.shape[0]
	cdef int n = E.shape[1]
	cdef int i, s
	with nogil:
		for s in prange(m, num_threads=t):
			for i in range(n):
				E[s,i] = E[s,i] - 2*f[s]
				E[s,i] = E[s,i]/sqrt(2*f[s]*(1 - f[s]))

# Add intercept to reconstructed allele frequencies
@boundscheck(False)
@wraparound(False)
cpdef updatePi(float[:,::1] P, float[::1] f, int t):
	cdef int m = P.shape[0]
	cdef int n = P.shape[1]
	cdef int i, s
	with nogil:
		for s in prange(m, num_threads=t):
			for i in range(n):
				P[s,i] = P[s,i] + 2*f[s]
				P[s,i] = P[s,i]/2
				P[s,i] = min(max(P[s,i], 1e-4), 1-(1e-4))
