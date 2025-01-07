# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
import numpy as np
cimport numpy as np
from cython.parallel import prange
from libc.math cimport sqrt

##### Cython functions for tree estimation #####
# Standardize individual allele frequencies
cpdef void standardizePi(float[:,::1] P, float[:,::1] PiNorm, double[::1] f, int t) \
		noexcept nogil:
	cdef:
		int m = P.shape[0]
		int n = P.shape[1]
		int i, j
	for j in prange(m, num_threads=t):
		for i in range(n):
			PiNorm[j,i] = P[j,i] - f[j]
			PiNorm[j,i] = PiNorm[j,i]/sqrt(f[j]*(1 - f[j]))

# Estimate distance matrix based on covariance matrix
cpdef void estimateDist(float[:,::1] C, float[:,::1] D) noexcept nogil:
	cdef:
		int n = C.shape[0]
		int i, j
	for i in range(n):
		for j in range(n):
			D[i,j] = max(0, C[i,i] + C[j,j] - 2*C[i,j])

# Estimate Q-matrix
cpdef void estimateQ(float[:,::1] D, float[:,::1] Q, float[::1] Dsum) noexcept nogil:
	cdef:
		int n = D.shape[0]
		int i, j
	for i in range(n):
		for j in range(n):
			Q[i,j] = (<float>(n) - 2)*D[i,j] - Dsum[i] - Dsum[j]

# New D-matrix
cpdef void updateD(float[:,::1] D0, float[:,::1] D, int pA, int pB) noexcept nogil:
	cdef:
		int n = D0.shape[0]
		int c = 0
		int i, j, d
	for i in range(n):
		if (i == pA) or (i == pB):
			continue
		else:
			d = 0
			for j in range(n):
				if (j == pA) or (j == pB):
					continue
				else:
					D[c,d] = D0[i,j]
					d = d + 1
			D[c,d] = max(0, 0.5*(D0[i, pA] + D0[i, pB] - D0[pA, pB]))
			D[d,c] = D[c,d]
			c = c + 1
