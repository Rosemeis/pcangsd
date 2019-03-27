import numpy as np
cimport numpy as np
from cython.parallel import prange
from cython import boundscheck, wraparound
from libc.math cimport log

##### Cython functions for EM inbreeding functions ######

# Update factor matrices
@boundscheck(False)
@wraparound(False)
cpdef updateF(float[:,::1] F, float[:,::1] A, float[:,::1] FB, int t):
	cdef int m = F.shape[0]
	cdef int K = F.shape[1]
	cdef int j, k
	for j in range(m):
		for k in range(K):
			F[j,k] = F[j,k]*A[j,k]/FB[j,k]
			F[j,k] = min(max(F[j,k], 1e-4), 1-(1e-4))

@boundscheck(False)
@wraparound(False)
cpdef updateQ(float[:,::1] Q, float[:,::1] A, float[:,::1] QB, float alpha, int t):
	cdef int n = Q.shape[0]
	cdef int K = Q.shape[1]
	cdef int i, k
	for i in range(n):
		for k in range(K):
			Q[i,k] = Q[i,k]*A[i,k]/(QB[i,k] + alpha)
			Q[i,k] = min(max(Q[i,k], 1e-4), 1-(1e-4))

# Log-likelihood
@boundscheck(False)
@wraparound(False)
cpdef loglike(float[:,::1] L, float[:,::1] Pi, float[::1] loglike_vec, int t):
	cdef int n = Pi.shape[0]
	cdef int m = Pi.shape[1]
	cdef int i, j
	cdef float like0, like1, like2
	with nogil:
		for i in prange(n, num_threads=t):
			for j in range(m):
				like0 = L[3*i,j]*(1 - Pi[i,j])*(1 - Pi[i,j])
				like1 = L[3*i+1,j]*2*Pi[i,j]*(1 - Pi[i,j])
				like2 = L[3*i+2,j]*Pi[i,j]*Pi[i,j]
				loglike_vec[i] += log(like0 + like1 + like2)

# Clip frequency matrix
@boundscheck(False)
@wraparound(False)
cpdef clipX(float[:,::1] X, int t):
	cdef int n = X.shape[0]
	cdef int m = X.shape[1]
	cdef int i, j
	with nogil:
		for i in prange(n, num_threads=t):
			for j in range(m):
				X[i,j] = min(max(X[i,j], 1e-4), 1-(1e-4))