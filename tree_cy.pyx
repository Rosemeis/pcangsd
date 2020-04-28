import numpy as np
cimport numpy as np
from cython.parallel import prange
from cython import boundscheck, wraparound
from libc.math cimport sqrt

##### Cython functions for tree construction ######

# Standardize IAF
@boundscheck(False)
@wraparound(False)
cpdef standardizePi(float[:,::1] Pi, float[::1] f, float[:,::1] PiNorm, int t):
	cdef int n = Pi.shape[0]
	cdef int m = Pi.shape[1]
	cdef int i, j
	with nogil:
		for i in prange(n, num_threads=t):
			for j in range(m):
				PiNorm[i,j] = Pi[i,j] - f[j]
				PiNorm[i,j] = PiNorm[i,j]/sqrt(f[j]*(1 - f[j]))

# Estimate distance matrix based on covariance matrix
@boundscheck(False)
@wraparound(False)
cpdef estimateDist(float[:,::1] Covar, float[:,::1] D):
	cdef int n = Covar.shape[0]
	cdef int i, j
	for i in range(n):
		for j in range(n):
			D[i,j] = max(0, Covar[i,i] + Covar[j,j] - 2*Covar[i,j])

# Estimate Q-matrix
@boundscheck(False)
@wraparound(False)
cpdef estimateQ(float[:,::1] D, float[:,::1] Q, float[:] Dsum):
	cdef int n = D.shape[0]
	cdef int i, j
	for i in range(n):
		for j in range(n):
			Q[i,j] = (n - 2)*D[i,j] - Dsum[i] - Dsum[j]

# New D-matrix
@boundscheck(False)
@wraparound(False)
cpdef updateD(float[:,::1] D0, float[:,::1] D, int pairA, int pairB):
	cdef int n = D0.shape[0]
	cdef int i, j, d
	cdef int c = 0
	for i in range(n):
		if (i == pairA) or (i == pairB):
			continue
		else:
			d = 0
			for j in range(n):
				if (j == pairA) or (j == pairB):
					continue
				else:
					D[c,d] = D0[i,j]
					d = d + 1
			D[c,d] = max(0, 0.5*(D0[i,pairA] + D0[i,pairB] - D0[pairA,pairB]))
			D[d,c] = D[c,d]
			c = c + 1