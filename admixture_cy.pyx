import numpy as np
cimport numpy as np
from cython.parallel import prange
from cython import boundscheck, wraparound
from libc.math cimport log, sqrt

##### Cython functions for admixture estimations ######

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

### Admixture selection scan
# Compute admixture based selection statistics
@boundscheck(False)
@wraparound(False)
cpdef admixFst(float[:,::1] F, float[::1] Qk, float[::1] Fst, int t):
	cdef int K = Qk.shape[0]
	cdef int m = F.shape[0]
	cdef int j, k
	cdef float Hs, Ht
	with nogil:
		for j in prange(m, num_threads=t):
			Hs = 0
			Ht = 0
			for k in range(K):
				Hs = Hs + Qk[k]*F[j,k]*(1 - F[j,k])
				Ht = Ht + Qk[k]*F[j,k]
			Ht = Ht*(1 - Ht)
			Fst[j] = 1 - Hs/Ht
			Fst[j] = min(max(Fst[j], 1e-4), 1-(1e-4))

### Admixture covariance matrix (K x K)
# Standardize ancestral allele frequencies
@boundscheck(False)
@wraparound(False)
cpdef admixNorm(float[:,::1] F, float[::1] f, float[:,::1] Fnorm, int t):
	cdef int m = F.shape[0]
	cdef int K = F.shape[1]
	cdef int j, k
	with nogil:
		for j in prange(m, num_threads=t):
			for k in range(K):
				Fnorm[j,k] = F[j,k] - f[j]
				Fnorm[j,k] = Fnorm[j,k]/sqrt(f[j]*(1 - f[j]))

### Admixture selection scan
# Center admixture proportions
@boundscheck(False)
@wraparound(False)
cpdef centerQ(float[:,::1] Q, float[::1] Qavg, float[:,::1] B, int t):
	cdef int n = Q.shape[0]
	cdef int K = Q.shape[1]
	cdef int i, k
	with nogil:
		for i in prange(n, num_threads=t):
			for k in range(K):
				B[i,k] = Q[i,k] - Qavg[k]

# Standardize IAG
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

# Estimate admixture selection scan
@boundscheck(False)
@wraparound(False)
cpdef estimateScan(float[:,::1] Pi, float[:] b, float[::1] f, float bCb, float[:] s, int t):
	cdef int n = Pi.shape[0]
	cdef int m = Pi.shape[1]
	cdef int i, j
	with nogil:
		for j in prange(m, num_threads=t):
			for i in range(n):
				s[j] = s[j] + (Pi[i,j] - f[j])*b[i]
			s[j] = s[j]**2
			s[j] = s[j]/(f[j]*(1 - f[j])*bCb)