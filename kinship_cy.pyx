import numpy as np
cimport numpy as np
from cython.parallel import prange
from cython import boundscheck, wraparound
from libc.math cimport sqrt

##### Cython functions for kinship.py ######

# Estimate diagonal entries of kinship matrix
@boundscheck(False)
@wraparound(False)
cpdef diagKinship(float[:,::1] L, float[:,::1] Pi, float[::1] dKin, int t):
	cdef int n = Pi.shape[0]
	cdef int m = Pi.shape[1]
	cdef int i, j
	cdef float num, dem, p0, p1, p2, pSum
	with nogil:
		for i in prange(n, num_threads=t):
			num = 0.0
			dem = 0.0
			for j in range(m):
				p0 = L[3*i,j]*(1 - Pi[i,j])*(1 - Pi[i,j])
				p1 = L[3*i+1,j]*2*Pi[i,j]*(1 - Pi[i,j])
				p2 = L[3*i+2,j]*Pi[i,j]*Pi[i,j]
				pSum = p0 + p1 + p2

				num = num + (0 - 2*Pi[i,j])*(0 - 2*Pi[i,j])*(p0/pSum)
				num = num + (1 - 2*Pi[i,j])*(1 - 2*Pi[i,j])*(p1/pSum)
				num = num + (2 - 2*Pi[i,j])*(2 - 2*Pi[i,j])*(p2/pSum)
				dem = dem + Pi[i,j]*(1 - Pi[i,j])
			dKin[i] = num/(4*dem)

# Prepare numerator matrix
@boundscheck(False)
@wraparound(False)
cpdef numeratorKin(float[:,::1] E, float[:,::1] Pi, int t):
	cdef int n = E.shape[0]
	cdef int m = E.shape[1]
	cdef int i, j
	with nogil:
		for i in prange(n, num_threads=t):
			for j in range(m):
				E[i,j] = E[i,j] - 2*Pi[i,j]

# Prepare denominator matrix
@boundscheck(False)
@wraparound(False)
cpdef denominatorKin(float[:,::1] E, float[:,::1] Pi, int t):
	cdef int n = E.shape[0]
	cdef int m = E.shape[1]
	cdef int i, j
	with nogil:
		for i in prange(n, num_threads=t):
			for j in range(m):
				E[i,j] = sqrt(Pi[i,j]*(1 - Pi[i,j]))