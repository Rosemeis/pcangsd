import numpy as np
cimport numpy as np
from cython.parallel import prange
from cython import boundscheck, wraparound
from libc.math cimport sqrt
from libc.stdlib cimport abort, malloc, free

##### Cython functions for covariance.py ######

# Update posterior expectations of the genotypes (Fumagalli method)
@boundscheck(False)
@wraparound(False)
cpdef updateFumagalli(float[:,::1] L, float[::1] f, float[:,::1] E, int t):
	cdef int n = L.shape[0]//3
	cdef int m = L.shape[1]
	cdef int i, j
	cdef float p0, p1, p2
	with nogil:
		for i in prange(n, num_threads=t):
			for j in range(m):
				p0 = L[3*i,j]*(1 - f[j])*(1 - f[j])
				p1 = L[3*i+1,j]*2*f[j]*(1 - f[j])
				p2 = L[3*i+2,j]*f[j]*f[j]

				# Update dosage
				E[i,j] = (p1 + 2*p2)/(p0 + p1 + p2)

# Update posterior expectations of the genotypes (PCAngsd method)
@boundscheck(False)
@wraparound(False)
cpdef updatePCAngsd(float[:,::1] L, float[:,::1] Pi, float[:,::1] E, int t):
	cdef int n = L.shape[0]//3
	cdef int m = L.shape[1]
	cdef int i, j
	cdef float p0, p1, p2
	with nogil:
		for j in prange(m, num_threads=t):
			for i in range(n):
				p0 = L[3*i,j]*(1 - Pi[i,j])*(1 - Pi[i,j])
				p1 = L[3*i+1,j]*2*Pi[i,j]*(1 - Pi[i,j])
				p2 = L[3*i+2,j]*Pi[i,j]*Pi[i,j]

				# Update dosage
				E[i,j] = (p1 + 2*p2)/(p0 + p1 + p2)

# Update posterior expectations of the genotypes including cov diagonal (Fumagalli method)
@boundscheck(False)
@wraparound(False)
cpdef covFumagalli(float[:,::1] L, float[::1] f, float[:,::1] E, float[::1] dCov, int t):
	cdef int n = L.shape[0]//3
	cdef int m = L.shape[1]
	cdef int i, j
	cdef float p0, p1, p2, pSum, temp
	with nogil:
		for i in prange(n, num_threads=t):
			dCov[i] = 0.0
			for j in range(m):
				p0 = L[3*i,j]*(1 - f[j])*(1 - f[j])
				p1 = L[3*i+1,j]*2*f[j]*(1 - f[j])
				p2 = L[3*i+2,j]*f[j]*f[j]
				pSum = p0 + p1 + p2

				# Update dosage and cov diagonal
				E[i,j] = (p1 + 2*p2)/pSum
				temp = (0 - 2*f[j])*(0 - 2*f[j])*(p0/pSum)
				temp = temp + (1 - 2*f[j])*(1 - 2*f[j])*(p1/pSum)
				temp = temp + (2 - 2*f[j])*(2 - 2*f[j])*(p2/pSum)
				dCov[i] = dCov[i] + temp/(2*f[j]*(1 - f[j]))
			dCov[i] = dCov[i]/m

# Update posterior expectations of the genotypes including cov diagonal (Fumagalli method)
@boundscheck(False)
@wraparound(False)
cpdef covPCAngsd(float[:,::1] L, float[::1] f, float[:,::1] Pi, float[:,::1] E, float[::1] dCov, int t):
	cdef int n = L.shape[0]//3
	cdef int m = L.shape[1]
	cdef int i, j
	cdef float p0, p1, p2, pSum, temp
	with nogil:
		for i in prange(n, num_threads=t):
			dCov[i] = 0.0
			for j in range(m):
				p0 = L[3*i,j]*(1 - Pi[i,j])*(1 - Pi[i,j])
				p1 = L[3*i+1,j]*2*Pi[i,j]*(1 - Pi[i,j])
				p2 = L[3*i+2,j]*Pi[i,j]*Pi[i,j]
				pSum = p0 + p1 + p2

				# Update dosage and cov diagonal
				E[i,j] = (p1 + 2*p2)/pSum
				temp = (0 - 2*f[j])*(0 - 2*f[j])*(p0/pSum)
				temp = temp + (1 - 2*f[j])*(1 - 2*f[j])*(p1/pSum)
				temp = temp + (2 - 2*f[j])*(2 - 2*f[j])*(p2/pSum)
				dCov[i] = dCov[i] + temp/(2*f[j]*(1 - f[j]))
			dCov[i] = dCov[i]/m

# Center posterior expectations of the genotype
@boundscheck(False)
@wraparound(False)
cpdef centerE(float[:,::1] E, float[::1] f, int t):
	cdef int n = E.shape[0]
	cdef int m = E.shape[1]
	cdef int i, j
	with nogil:
		for i in prange(n, num_threads=t):
			for j in range(m):
				E[i,j] = E[i,j] - 2*f[j]

# Standardize posterior expectations of the genotype
@boundscheck(False)
@wraparound(False)
cpdef standardizeE(float[:,::1] E, float[::1] f, int t):
	cdef int n = E.shape[0]
	cdef int m = E.shape[1]
	cdef int i, j
	with nogil:
		for i in prange(n, num_threads=t):
			for j in range(m):
				E[i,j] = E[i,j] - 2*f[j]
				E[i,j] = E[i,j]/sqrt(2*f[j]*(1 - f[j]))

# Add intercept to reconstructed allele frequencies
@boundscheck(False)
@wraparound(False)
cpdef updatePi(float[:,::1] Pi, float[::1] f, int t):
	cdef int n = Pi.shape[0]
	cdef int m = Pi.shape[1]
	cdef int i, j
	with nogil:
		for i in prange(n, num_threads=t):
			for j in range(m):
				Pi[i,j] = Pi[i,j] + 2*f[j]
				Pi[i,j] = Pi[i,j]/2
				Pi[i,j] = min(max(Pi[i,j], 1e-4), 1-(1e-4))

# RMSE with sign checking
@boundscheck(False)
@wraparound(False)
cpdef rmse2d_eig(float[:,:] A, float[:,:] B):
	cdef int n = A.shape[0]
	cdef int m = A.shape[1]
	cdef int i, j
	cdef float res = 0.0
	for i in range(n):
		for j in range(m):
			if (A[i,j] > 0) & (B[i,j] > 0):
				res += (A[i,j] - B[i,j])*(A[i,j] - B[i,j])
			elif (A[i,j] < 0) & (B[i,j] < 0):
				res += (A[i,j] - B[i,j])*(A[i,j] - B[i,j])
			else:
				res += (A[i,j] + B[i,j])*(A[i,j] + B[i,j])
	res /= (m*n)
	return sqrt(res)