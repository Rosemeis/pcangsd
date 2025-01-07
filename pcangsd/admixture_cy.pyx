# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
import numpy as np
cimport numpy as np
from cython.parallel import prange
from libc.math cimport log

##### Cython functions for admixture estimation #####
# Update factor matrices
cpdef void updateF(float[:,::1] F, float[:,::1] A, float[:,::1] FB) noexcept nogil:
    cdef:
        int m = F.shape[0]
        int K = F.shape[1]
        int j, k
    for j in range(m):
        for k in range(K):
            F[j,k] = F[j,k]*A[j,k]/FB[j,k]
            F[j,k] = min(max(F[j,k], 1e-4), 1-(1e-4))

cpdef void updateQ(float[:,::1] Q, float[:,::1] A, float[:,::1] QB, float alpha) \
        noexcept nogil:
    cdef:
        int n = Q.shape[0]
        int K = Q.shape[1]
        int i, k
    for i in range(n):
        for k in range(K):
            Q[i,k] = Q[i,k]*A[i,k]/(QB[i,k] + alpha)
            Q[i,k] = min(max(Q[i,k], 1e-4), 1-(1e-4))

# Log-likelihood
cpdef void loglike(float[:,::1] L, float[:,::1] X, double[::1] logvec, int t) \
        noexcept nogil:
    cdef:
        int m = X.shape[0]
        int n = X.shape[1]
        int i, j
        double l0, l1, l2
    for j in prange(m, num_threads=t):
        for i in range(n):
            l0 = L[j,2*i+0]*(1 - X[j,i])*(1 - X[j,i])
            l1 = L[j,2*i+1]*2*X[j,i]*(1 - X[j,i])
            l2 = (1.0 - L[j,2*i+0] - L[j,2*i+1])*X[j,i]*X[j,i]
            logvec[j] = logvec[j] + log(l0 + l1 + l2)
