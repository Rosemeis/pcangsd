import numpy as np
cimport numpy as np
from cython.parallel import prange
from cython import boundscheck, wraparound
from libc.math cimport sqrt

##### Cython functions for tree estimation #####
# Standardize individual allele frequencies
@boundscheck(False)
@wraparound(False)
cpdef standardizePi(float[:,::1] P, float[::1] f, float[:,::1] PiNorm, int t):
    cdef int m = P.shape[0]
    cdef int n = P.shape[1]
    cdef int i, s
    with nogil:
        for s in prange(m, num_threads=t):
            for i in range(n):
                PiNorm[s,i] = P[s,i] - f[s]
                PiNorm[s,i] = PiNorm[s,i]/sqrt(f[s]*(1 - f[s]))

# Estimate distance matrix based on covariance matrix
@boundscheck(False)
@wraparound(False)
cpdef estimateDist(float[:,::1] C, float[:,::1] D):
    cdef int n = C.shape[0]
    cdef int i, j
    for i in range(n):
        for j in range(n):
            D[i,j] = max(0, C[i,i] + C[j,j] - 2*C[i,j])

# Estimate Q-matrix
@boundscheck(False)
@wraparound(False)
cpdef estimateQ(float[:,::1] D, float[:,::1] Q, float[::1] Dsum):
    cdef int n = D.shape[0]
    cdef int i, j
    for i in range(n):
        for j in range(n):
            Q[i,j] = (<float>(n) - 2)*D[i,j] - Dsum[i] - Dsum[j]

# New D-matrix
@boundscheck(False)
@wraparound(False)
cpdef updateD(float[:,::1] D0, float[:,::1] D, int pA, int pB):
    cdef int n = D0.shape[0]
    cdef int i, j, d
    cdef int c = 0
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
