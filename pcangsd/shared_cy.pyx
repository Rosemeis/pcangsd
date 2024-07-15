# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
import numpy as np
cimport numpy as np
from cython.parallel import prange
from libc.math cimport sqrt

##### Shared Cython functions #####
# EM MAF update
cpdef void emMAF_update(float[:,::1] L, double[::1] f, int t) noexcept nogil:
	cdef:
		int m = L.shape[0]
		int n = L.shape[1]//2
		int i, j
		double s = 1/(2*<double>n)
		double tmp, p0, p1, p2
	for j in prange(m, num_threads=t):
		tmp = 0.0
		for i in range(n):
			p0 = L[j,2*i+0]*(1.0-f[j])*(1.0-f[j])
			p1 = L[j,2*i+1]*2.0*f[j]*(1.0-f[j])
			p2 = (1.0 - L[j,2*i+0] - L[j,2*i+1])*f[j]*f[j]
			tmp = tmp + (p1 + 2.0*p2)/(2.0*(p0 + p1 + p2))
		f[j] = tmp*s

# EM MAF accelerated update
cpdef void emMAF_accel(float[:,::1] L, double[::1] f, double[::1] f_new, \
		double[::1] d, int t) noexcept nogil:
	cdef:
		int m = L.shape[0]
		int n = L.shape[1]//2
		int i, j
		double s = 1/(2*<double>n)
		double tmp, p0, p1, p2
	for j in prange(m, num_threads=t):
		tmp = 0.0
		for i in range(n):
			p0 = L[j,2*i+0]*(1.0-f[j])*(1.0-f[j])
			p1 = L[j,2*i+1]*2.0*f[j]*(1.0-f[j])
			p2 = (1.0 - L[j,2*i+0] - L[j,2*i+1])*f[j]*f[j]
			tmp = tmp + (p1 + 2.0*p2)/(2.0*(p0 + p1 + p2))
		f_new[j] = tmp*s
		d[j] = f_new[j] - f[j]

# Vector subtraction
cpdef void vecMinus(double[::1] a, double[::1] b, double[::1] c) noexcept nogil:
	cdef:
		int m = a.shape[0]
		int j
	for j in range(m):
		c[j] = a[j] - b[j]

# Vector sum of squares
cpdef double vecSumSquare(double[::1] a) nogil:
	cdef:
		int m = a.shape[0]
		int j
		double res = 0.0
	for j in range(m):
		res += a[j]*a[j]
	return res

# Alpha update SqS3
cpdef void vecUpdate(double[::1] a, double[::1] a0, double[::1] d1, double[::1] d3, \
		double alpha) noexcept nogil:
	cdef:
		int m = a.shape[0]
		int j
	for j in range(m):
		a[j] = a0[j] - 2*alpha*d1[j] + alpha*alpha*d3[j]

# Root mean squared error (1D)
cpdef double rmse1d(double[::1] a, double[::1] b) noexcept nogil:
	cdef:
		int n = a.shape[0]
		int i
		double res = 0.0
	for i in range(n):
		res = res + (a[i] - b[i])*(a[i] - b[i])
	res = res/(<double>n)
	return sqrt(res)

# Root mean squared error (2D)
cpdef double rmse2d(float[:,:] A, float[:,:] B) noexcept nogil:
	cdef:
		int n = A.shape[0]
		int m = A.shape[1]
		int i, j
		double res = 0.0
	for i in range(n):
		for j in range(m):
			res = res + (A[i,j] - B[i,j])*(A[i,j] - B[i,j])
	res = res/(<double>(n*m))
	return sqrt(res)

# Frobenius error
cpdef double frobenius(float[:,::1] A, float[:,::1] B) noexcept nogil:
	cdef:
		int n = A.shape[0]
		int m = A.shape[1]
		int i, j
		double res = 0.0
	for i in range(n):
		for j in range(m):
			res += (A[i,j] - B[i,j])*(A[i,j] - B[i,j])
	return sqrt(res)

# Frobenius error - threaded
cpdef void frobeniusThread(float[:,::1] A, float[:,::1] B, float[::1] res_vec, \
		int t) noexcept nogil:
	cdef:
		int n = A.shape[0]
		int m = A.shape[1]
		int i, j
	for i in prange(n, num_threads=t):
		for j in range(m):
			res_vec[i] = res_vec[i] + (A[i,j] - B[i,j])*(A[i,j] - B[i,j])

# FastPCA selection scan
cpdef void computeD(float[:,:] U, float[:,::1] D) noexcept nogil:
	cdef:
		int m = U.shape[0]
		int K = U.shape[1]
		int j, k
	for j in range(m):
		for k in range(K):
			D[j,k] = (U[j,k]*U[j,k])*<float>(m)

# pcadapt selection scan
cpdef void computeZ(float[:,::1] E, float[:,::1] B, float[:,:] Vt, float[:,::1] Z) \
		noexcept nogil:
	cdef:
		int m = E.shape[0]
		int n = E.shape[1]
		int K = Vt.shape[0]
		int i, j, k
		double rec, res
	for j in range(m):
		res = 0.0
		for i in range(n):
			rec = 0.0
			for k in range(K):
				rec = rec + Vt[k,i]*B[j,k]
			res = res + (E[j,i] - rec)*(E[j,i] - rec)
		res = sqrt(res/<double>(n-K))
		if res > 0:
			for k in range(K):
				Z[j,k] = B[j,k]/res
		else:
			for k in range(K):
				Z[j,k] = 0.0

# Genotype calling
cpdef void geno(float[:,::1] L, float[:,::1] P, signed char[:,::1] G, \
		double delta, int t) noexcept nogil:
	cdef:
		int m = P.shape[0]
		int n = P.shape[1]
		int i, j
		double p0, p1, p2, pSum
	for j in prange(m, num_threads=t):
		for i in range(n):
			p0 = L[j,2*i+0]*(1.0-P[j,i])*(1.0-P[j,i])
			p1 = L[j,2*i+1]*2.0*P[j,i]*(1.0-P[j,i])
			p2 = (1.0 - L[j,2*i+0] - L[j,2*i+1])*P[j,i]*P[j,i]
			pSum = p0 + p1 + p2

			# Call posterior maximum
			if (p0 > p1) & (p0 > p2):
				if (p0/pSum > delta):
					G[j,i] = 0
				else:
					G[j,i] = -9
			elif (p1 > p2):
				if (p1/pSum > delta):
					G[j,i] = 1
				else:
					G[j,i] = -9
			else:
				if (p2/pSum > delta):
					G[j,i] = 2
				else:
					G[j,i] = -9

# Genotype calling (inbreeding)
cpdef void genoInbreed(float[:,::1] L, float[:,::1] P, double[::1] F, \
		signed char[:,::1] G, double delta, int t) noexcept nogil:
	cdef:
		int m = P.shape[0]
		int n = P.shape[1]
		int i, j
		double p0, p1, p2, pSum
	for j in prange(m, num_threads=t):
		for i in range(n):
			p0 = L[j,2*i+0]*((1.0-P[j,i])*(1.0-P[j,i]) + \
				P[j,i]*(1.0-P[j,i])*F[i])
			p1 = L[j,2*i+1]*2.0*P[j,i]*(1.0-P[j,i])
			p2 = (1.0 - L[j,2*i+0] - L[j,2*i+1])*(P[j,i]*P[j,i] + \
				P[j,i]*(1.0-P[j,i])*F[i])
			pSum = p0 + p1 + p2

			# Call maximum posterior
			if (p0 > p1) & (p0 > p2):
				if (p0/pSum > delta):
					G[j,i] = 0
				else:
					G[j,i] = -9
			elif (p1 > p2):
				if (p1/pSum > delta):
					G[j,i] = 1
				else:
					G[j,i] = -9
			else:
				if (p2/pSum > delta):
					G[j,i] = 2
				else:
					G[j,i] = -9

# Create fake frequencies
cpdef void freqs(float[:,::1] P, double[::1] f, int t) noexcept nogil:
	cdef:
		int m = P.shape[0]
		int n = P.shape[1]
		int i, j
	for j in prange(m, num_threads=t):
		for i in range(n):
			P[j,i] = f[j]
