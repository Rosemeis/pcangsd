import numpy as np
cimport numpy as np
from cython.parallel import prange
from cython import boundscheck, wraparound
from libc.math cimport sqrt

##### Shared Cython functions #####

# Root mean squared error (1D)
@boundscheck(False)
@wraparound(False)
cpdef rmse1d(float[::1] A, float[::1] B):
	cdef int n = A.shape[0]
	cdef int i
	cdef float res = 0.0
	for i in range(n):
		res += (A[i] - B[i])*(A[i] - B[i])
	res /= n
	return sqrt(res)

# Root mean squared error (2D)
@boundscheck(False)
@wraparound(False)
cpdef rmse2d(float[:,::1] A, float[:,::1] B):
	cdef int n = A.shape[0]
	cdef int m = A.shape[1]
	cdef int i, j
	cdef float res = 0.0
	for i in range(n):
		for j in range(m):
			res += (A[i,j] - B[i,j])*(A[i,j] - B[i,j])
	res /= (m*n)
	return sqrt(res)

# Frobenius norm (2D)
@boundscheck(False)
@wraparound(False)
cpdef frobenius(float[:,:] A, float[:,:] B):
	cdef int n = A.shape[0]
	cdef int m = A.shape[1]
	cdef int i, j
	cdef float res = 0.0
	for i in range(n):
		for j in range(m):
			res += (A[i,j] - B[i,j])*(A[i,j] - B[i,j])
	return sqrt(res)

##### Cython functions for emMaf.py #####
@boundscheck(False)
@wraparound(False)
cpdef emMaf_update(float[:,::1] L, float[::1] f, float[::1] newF, int t):
	cdef int n = L.shape[0]//3
	cdef int m = L.shape[1]
	cdef int i, j
	cdef float p0, p1, p2
	with nogil:
		for j in prange(m, num_threads=t):
			newF[j] = 0.0
			for i in range(n):
				p0 = L[3*i,j]*(1 - f[j])*(1 - f[j])
				p1 = L[3*i+1,j]*2*f[j]*(1 - f[j])
				p2 = L[3*i+2,j]*f[j]*f[j]
				newF[j] = newF[j] + (p1 + 2*p2)/(2*(p0 + p1 + p2))
			f[j] = newF[j]/n

##### Cython functions for selection.py #####
@boundscheck(False)
@wraparound(False)
cpdef computeD(float[:,:] U, float[:,:] Dsquared):
	cdef int k = U.shape[0]
	cdef int m = U.shape[1]
	cdef int i, j
	for j in range(m):
		for i in range(k):
			Dsquared[j,i] = (U[i,j]**2)*m

##### Cython functions for saving posterior genotype probabilities #####
@boundscheck(False)
@wraparound(False)
cpdef computePostPi(float[:,::1] L, float[:,::1] Pi, int t):
	cdef int n = Pi.shape[0]
	cdef int m = Pi.shape[1]
	cdef int i, j
	cdef float p0, p1, p2, pSum
	with nogil:
		for i in prange(n, num_threads=t):
			for j in range(m):
				p0 = L[3*i,j]*(1 - Pi[i,j])*(1 - Pi[i,j])
				p1 = L[3*i+1,j]*2*Pi[i,j]*(1 - Pi[i,j])
				p2 = L[3*i+2,j]*Pi[i,j]*Pi[i,j]
				pSum = p0 + p1 + p2

				# Update L
				L[3*i,j] = p0/pSum
				L[3*i+1,j] = p1/pSum
				L[3*i+2,j] = p2/pSum

@boundscheck(False)
@wraparound(False)
cpdef computePostF(float[:,::1] L, float[::1] f, int t):
	cdef int n = L.shape[0]//3
	cdef int m = L.shape[1]
	cdef int i, j
	cdef float p0, p1, p2, pSum
	with nogil:
		for i in prange(n, num_threads=t):
			for j in range(m):
				p0 = L[3*i,j]*(1 - f[j])*(1 - f[j])
				p1 = L[3*i+1,j]*2*f[j]*(1 - f[j])
				p2 = L[3*i+2,j]*f[j]*f[j]
				pSum = p0 + p1 + p2

				# Update L
				L[3*i,j] = p0/pSum
				L[3*i+1,j] = p1/pSum
				L[3*i+2,j] = p2/pSum		