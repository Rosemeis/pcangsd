import numpy as np
cimport numpy as np
from cython.parallel import prange
from cython import boundscheck, wraparound
from libc.math cimport sqrt

##### Shared Cython functions #####
# EM MAF update
@boundscheck(False)
@wraparound(False)
cpdef emMAF_update(float[:,::1] L, float[::1] f, int t):
    cdef int m = L.shape[0]
    cdef int n = L.shape[1]//2
    cdef int i, s
    cdef float tmp, p0, p1, p2
    with nogil:
        for s in prange(m, num_threads=t):
            tmp = 0.0
            for i in range(n):
                p0 = L[s, 2*i+0]*(1 - f[s])*(1 - f[s])
                p1 = L[s, 2*i+1]*2*f[s]*(1 - f[s])
                p2 = (1.0 - L[s, 2*i+0] - L[s, 2*i+1])*f[s]*f[s]
                tmp = tmp + (p1 + 2*p2)/(2*(p0 + p1 + p2))
            f[s] = tmp/(<float>n)

# Root mean squared error (1D)
@boundscheck(False)
@wraparound(False)
cpdef rmse1d(float[::1] v1, float[::1] v2):
    cdef int n = v1.shape[0]
    cdef int i
    cdef float res = 0.0
    for i in range(n):
        res = res + (v1[i] - v2[i])*(v1[i] - v2[i])
    res = res/(<float>n)
    return sqrt(res)

# Root mean squared error (2D)
@boundscheck(False)
@wraparound(False)
cpdef rmse2d(float[:,:] M1, float[:,:] M2):
    cdef int n = M1.shape[0]
    cdef int m = M1.shape[1]
    cdef int i, j
    cdef float res = 0.0
    for i in range(n):
        for j in range(m):
            res = res + (M1[i,j] - M2[i,j])*(M1[i,j] - M2[i,j])
    res = res/(<float>(n*m))
    return sqrt(res)

# Frobenius error
@boundscheck(False)
@wraparound(False)
cpdef frobenius(float[:,::1] A, float[:,::1] B):
    cdef int n = A.shape[0]
    cdef int m = A.shape[1]
    cdef int i, j
    cdef float res = 0.0
    for i in range(n):
        for j in range(m):
            res += (A[i,j] - B[i,j])*(A[i,j] - B[i,j])
    return sqrt(res)

# Frobenius error - threaded
@boundscheck(False)
@wraparound(False)
cpdef frobeniusThread(float[:,::1] A, float[:,::1] B, float[::1] res_vec, int t):
    cdef int n = A.shape[0]
    cdef int m = A.shape[1]
    cdef int i, j
    with nogil:
        for i in prange(n, num_threads=t):
            for j in range(m):
                res_vec[i] = res_vec[i] + (A[i,j] - B[i,j])*(A[i,j] - B[i,j])

# FastPCA selection scan
@boundscheck(False)
@wraparound(False)
cpdef computeD(float[:,:] U, float[:,::1] D):
    cdef int m = U.shape[0]
    cdef int K = U.shape[1]
    cdef int k, s
    for s in range(m):
        for k in range(K):
            D[s,k] = (U[s,k]*U[s,k])*<float>(m)

# pcadapt selection scan
@boundscheck(False)
@wraparound(False)
cpdef computeZ(float[:,::1] E, float[:,::1] B, float[:,:] Vt, float[:,::1] Z):
    cdef int m = E.shape[0]
    cdef int n = E.shape[1]
    cdef int K = Vt.shape[0]
    cdef int i, k, s
    cdef float rec, res
    for s in range(m):
        res = 0.0
        for i in range(n):
            rec = 0.0
            for k in range(K):
                rec = rec + Vt[k,i]*B[s,k]
            res = res + (E[s,i] - rec)*(E[s,i] - rec)
        res = sqrt(res/<float>(n-K))
        if res > 0:
            for k in range(K):
                Z[s,k] = B[s,k]/res
        else:
            for k in range(K):
                Z[s,k] = 0.0

# Genotype calling
@boundscheck(False)
@wraparound(False)
cpdef geno(float[:,::1] L, float[:,::1] P, signed char[:,::1] G, \
            float delta, int t):
    cdef int m = P.shape[0]
    cdef int n = P.shape[1]
    cdef int i, s
    cdef float p0, p1, p2, pSum
    with nogil:
        for s in prange(m, num_threads=t):
            for i in range(n):
                p0 = L[s,2*i+0]*(1 - P[s,i])*(1 - P[s,i])
                p1 = L[s,2*i+1]*2*P[s,i]*(1 - P[s,i])
                p2 = (1.0 - L[s, 2*i+0] - L[s, 2*i+1])*P[s,i]*P[s,i]
                pSum = p0 + p1 + p2

                # Call posterior maximum
                if (p0 > p1) & (p0 > p2):
                    if (p0/pSum > delta):
                        G[s,i] = 0
                    else:
                        G[s,i] = -9
                elif (p1 > p2):
                    if (p1/pSum > delta):
                        G[s,i] = 1
                    else:
                        G[s,i] = -9
                else:
                    if (p2/pSum > delta):
                        G[s,i] = 2
                    else:
                        G[s,i] = -9

# Genotype calling (inbreeding)
@boundscheck(False)
@wraparound(False)
cpdef genoInbreed(float[:,::1] L, float[:,::1] P, float[::1] F, \
                    signed char[:,::1] G, float delta, int t):
    cdef int m = P.shape[0]
    cdef int n = P.shape[1]
    cdef int i, s
    cdef float p0, p1, p2, pSum
    with nogil:
        for s in prange(m, num_threads=t):
            for i in range(n):
                p0 = L[s,2*i+0]*((1 - P[s,i])*(1 - P[s,i]) + \
                        P[s,i]*(1 - P[s,i])*F[i])
                p1 = L[s,2*i+1]*2*P[s,i]*(1 - P[s,i])
                p2 = (1.0 - L[s, 2*i+0] - L[s, 2*i+1])*(P[s,i]*P[s,i] + \
                        P[s,i]*(1 - P[s,i])*F[i])
                pSum = p0 + p1 + p2

                # Call maximum posterior
                if (p0 > p1) & (p0 > p2):
                    if (p0/pSum > delta):
                        G[s,i] = 0
                    else:
                        G[s,i] = -9
                elif (p1 > p2):
                    if (p1/pSum > delta):
                        G[s,i] = 1
                    else:
                        G[s,i] = -9
                else:
                    if (p2/pSum > delta):
                        G[s,i] = 2
                    else:
                        G[s,i] = -9
