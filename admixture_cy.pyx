import numpy as np
cimport numpy as np
from cython.parallel import prange
from cython import boundscheck, wraparound
from libc.math cimport log

##### Cython functions for admixture estimation #####
# Update factor matrices
@boundscheck(False)
@wraparound(False)
cpdef updateF(float[:,::1] F, float[:,::1] A, float[:,::1] FB):
    cdef int m = F.shape[0]
    cdef int K = F.shape[1]
    cdef int s, k
    for s in range(m):
        for k in range(K):
            F[s,k] = F[s,k]*A[s,k]/FB[s,k]
            F[s,k] = min(max(F[s,k], 1e-4), 1-(1e-4))

@boundscheck(False)
@wraparound(False)
cpdef updateQ(float[:,::1] Q, float[:,::1] A, float[:,::1] QB, float alpha):
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
cpdef loglike(float[:,::1] L, float[:,::1] X, float[::1] loglike_vec, int t):
    cdef int m = X.shape[0]
    cdef int n = X.shape[1]
    cdef int i, s
    cdef float like0, like1, like2
    with nogil:
        for s in prange(m, num_threads=t):
            for i in range(n):
                like0 = L[s,2*i+0]*(1 - X[s,i])*(1 - X[s,i])
                like1 = L[s,2*i+1]*2*X[s,i]*(1 - X[s,i])
                like2 = (1.0 - L[s,2*i+0] - L[s,2*i+1])*X[s,i]*X[s,i]
                loglike_vec[s] = loglike_vec[s] + log(like0 + like1 + like2)
