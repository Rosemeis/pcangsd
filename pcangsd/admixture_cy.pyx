# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
cimport numpy as np
from cython.parallel import prange
from libc.math cimport log

##### Cython functions for admixture estimation #####
# Update factor matrices
cpdef void updateF(float[:,::1] F, const float[:,::1] A, const float[:,::1] FB) \
        noexcept nogil:
    cdef:
        size_t M = F.shape[0]
        size_t K = F.shape[1]
        size_t j, k
    for j in prange(M):
        for k in range(K):
            F[j,k] = F[j,k]*A[j,k]/FB[j,k]
            F[j,k] = min(max(F[j,k], 1e-4), 1.0-(1e-4))

cpdef void updateQ(float[:,::1] Q, const float[:,::1] A, const float[:,::1] QB, \
        const float alpha) noexcept nogil:
    cdef:
        size_t N = Q.shape[0]
        size_t K = Q.shape[1]
        size_t i, k
        double sumQ, valQ
    for i in range(N):
        sumQ = 0.0
        for k in range(K):
            valQ = Q[i,k]*A[i,k]/(QB[i,k] + alpha)
            valQ = min(max(valQ, 1e-4), 1.0-(1e-4))
            sumQ += valQ
            Q[i,k] = valQ
        sumQ = 1.0/sumQ
        for k in range(K):
            Q[i,k] *= sumQ

# Log-likelihood
cpdef double loglike(const float[:,::1] L, const float[:,::1] X) noexcept nogil:
    cdef:
        size_t M = X.shape[0]
        size_t N = X.shape[1]
        size_t i, j
        double l0, l1, l2
        double ll = 0.0
    for j in prange(M):
        for i in range(N):
            l0 = L[j,2*i]*(1.0 - X[j,i])*(1.0 - X[j,i])
            l1 = L[j,2*i+1]*2.0*X[j,i]*(1 - X[j,i])
            l2 = (1.0 - L[j,2*i] - L[j,2*i+1])*X[j,i]*X[j,i]
            ll += log(l0 + l1 + l2)
    return ll
