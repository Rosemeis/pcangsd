# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
cimport numpy as np
from cython.parallel import prange
from libc.math cimport sqrt

##### Cython functions for tree estimation #####
# Standardize individual allele frequencies
cpdef void standardizePi(const float[:,::1] P, float[:,::1] Pi, const double[::1] f) \
		noexcept nogil:
	cdef:
		size_t M = P.shape[0]
		size_t N = P.shape[1]
		size_t i, j
		float fj, dj
	for j in prange(M):
		fj = f[j]
		dj = 1.0/sqrt(fj*(1.0 - fj))
		for i in range(N):
			Pi[j,i] = (P[j,i] - fj)*dj

# Estimate distance matrix based on covariance matrix
cpdef void estimateD(const float[:,::1] C, float[:,::1] D) noexcept nogil:
	cdef:
		size_t N = C.shape[0]
		size_t i, j
		float ci
	for i in range(N):
		ci = C[i,i]
		for j in range(N):
			D[i,j] = max(0, ci + C[j,j] - 2.0*C[i,j])

# Estimate Q-matrix
cpdef void estimateQ(const float[:,::1] D, float[:,::1] Q, const float[::1] D_sum) \
		noexcept nogil:
	cdef:
		size_t N = D.shape[0]
		size_t i, j
		float di
	for i in range(N):
		di = D_sum[i]
		for j in range(N):
			Q[i,j] = (<float>(N) - 2.0)*D[i,j] - di - D_sum[j]

# Update new distance matrix
cpdef void updateD(const float[:,::1] D0, float[:,::1] D, const size_t pA, \
		const size_t pB) noexcept nogil:
	cdef:
		size_t N = D0.shape[0]
		size_t c = 0
		size_t i, j, d
	for i in range(N):
		if (i == pA) or (i == pB):
			continue
		else:
			d = 0
			for j in range(N):
				if (j == pA) or (j == pB):
					continue
				else:
					D[c,d] = D0[i,j]
					d += 1
			D[c,d] = max(0, 0.5*(D0[i, pA] + D0[i, pB] - D0[pA, pB]))
			D[d,c] = D[c,d]
			c += 1
