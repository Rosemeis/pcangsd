# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
cimport numpy as np
from cython.parallel import prange
from libc.math cimport sqrt

##### Shared Cython functions #####
# Inline function
cdef inline double computeC(const double* x0, const double* x1, const double* x2, \
		const size_t I) noexcept nogil:
	cdef:
		size_t i
		double sum1 = 0.0
		double sum2 = 0.0
		double u, v
	for i in prange(I):
		u = x1[i] - x0[i]
		v = x2[i] - x1[i] - u
		sum1 += u*u
		sum2 += u*v
	return min(max(-(sum1/sum2), 1.0), 256.0)

# EM MAF update
cpdef void emMAF_update(const float[:,::1] L, double[::1] f) noexcept nogil:
	cdef:
		size_t M = L.shape[0]
		size_t N = L.shape[1]//2
		size_t i, j
		double s = 1.0/(2.0*<double>N)
		double fj, p0, p1, p2, tmp
	for j in prange(M):
		fj = f[j]
		tmp = 0.0
		for i in range(N):
			p0 = L[j,2*i]*(1.0 - fj)*(1.0 - fj)
			p1 = L[j,2*i+1]*2.0*fj*(1.0 - fj)
			p2 = (1.0 - L[j,2*i] - L[j,2*i+1])*fj*fj
			tmp = tmp + (p1 + 2.0*p2)/(p0 + p1 + p2)
		f[j] = tmp*s

# EM MAF accelerated update
cpdef void emMAF_accel(const float[:,::1] L, const double[::1] f, double[::1] f_new) \
		noexcept nogil:
	cdef:
		size_t M = L.shape[0]
		size_t N = L.shape[1]//2
		size_t i, j
		double s = 1.0/(2.0*<double>N)
		double fj, p0, p1, p2, tmp
	for j in prange(M):
		fj = f[j]
		tmp = 0.0
		for i in range(N):
			p0 = L[j,2*i]*(1.0 - fj)*(1.0 - fj)
			p1 = L[j,2*i+1]*2.0*fj*(1.0 - fj)
			p2 = (1.0 - L[j,2*i] - L[j,2*i+1])*fj*fj
			tmp = tmp + (p1 + 2.0*p2)/(p0 + p1 + p2)
		f_new[j] = tmp*s

# EM MAF QN jump
cpdef void emMAF_alpha(double[::1] f0, const double[::1] f1, const double[::1] f2) \
		noexcept nogil:
	cdef:
		size_t M = f0.shape[0]
		size_t j
		double c1, c2
	c1 = computeC(&f0[0], &f1[0], &f2[0], M)
	c2 = 1.0 - c1
	for j in prange(M):
		f0[j] = min(max(c2*f1[j] + c1*f2[j], 1e-5), 1.0-(1e-5))

# Inbreeding QN jump
cpdef void inbreed_alpha(double[::1] F0, const double[::1] F1, const double[::1] F2) \
		noexcept nogil:
	cdef:
		size_t I = F0.shape[0]
		size_t i
		double c1, c2
	c1 = computeC(&F0[0], &F1[0], &F2[0], I)
	c2 = 1.0 - c1
	for i in prange(I):
		F0[i] = min(max(c2*F1[i] + c1*F2[i], -1.0), 1.0)

# Root mean squared error (1D)
cpdef double rmse1d(const double[::1] a, const double[::1] b) noexcept nogil:
	cdef:
		size_t N = a.shape[0]
		size_t i
		double res = 0.0
	for i in range(N):
		res += (a[i] - b[i])*(a[i] - b[i])
	return sqrt(res/(<double>N))

# Root mean squared error (2D)
cpdef double rmse2d(const float[:,::1] A, const float[:,::1] B) noexcept nogil:
	cdef:
		size_t N = A.shape[0]
		size_t M = A.shape[1]
		size_t i, j
		double res = 0.0
	for i in range(N):
		for j in range(M):
			res += (A[i,j] - B[i,j])*(A[i,j] - B[i,j])
	return sqrt(res/(<double>(N*M)))

# Frobenius error
cpdef double frobenius(const float[:,::1] A, const float[:,::1] B) noexcept nogil:
	cdef:
		size_t N = A.shape[0]
		size_t M = A.shape[1]
		size_t i, j
		double res = 0.0
	for i in range(N):
		for j in range(M):
			res += (A[i,j] - B[i,j])*(A[i,j] - B[i,j])
	return sqrt(res)

# Frobenius error (multi-threaded)
cpdef double frobeniusMulti(const float[:,::1] A, const float[:,::1] B) noexcept nogil:
	cdef:
		size_t N = A.shape[0]
		size_t M = A.shape[1]
		size_t i, j
		double res = 0.0
	for i in prange(N):
		for j in range(M):
			res += (A[i,j] - B[i,j])*(A[i,j] - B[i,j])
	return sqrt(res)

# FastPCA selection scan
cpdef void computeD(const float[:,::1] U, float[:,::1] D) noexcept nogil:
	cdef:
		size_t M = U.shape[0]
		size_t K = U.shape[1]
		size_t j, k
	for j in prange(M):
		for k in range(K):
			D[j,k] = (U[j,k]*U[j,k])*<float>(M)

# pcadapt selection scan
cpdef void computeZ(const float[:,::1] E, const float[:,::1] B, const float[:,::1] V, \
		float[:,::1] Z) noexcept nogil:
	cdef:
		size_t M = E.shape[0]
		size_t N = E.shape[1]
		size_t K = Z.shape[1]
		size_t i, j, k
		double rec, res
	for j in range(M):
		res = 0.0
		for i in range(N):
			rec = 0.0
			for k in range(K):
				rec += V[i,k]*B[j,k]
			res += (E[j,i] - rec)*(E[j,i] - rec)
		if res > 0.0:
			res = 1.0/sqrt(res/<double>(N-K))
			for k in range(K):
				Z[j,k] = B[j,k]*res
		else:
			for k in range(K):
				Z[j,k] = 0.0

# Genotype posteriors based on individal allele frequencies
cpdef void post(const float[:,::1] L, const float[:,::1] P, float[:,::1] G) \
		noexcept nogil:
	cdef:
		size_t M = P.shape[0]
		size_t N = P.shape[1]
		size_t i, j
		double p0, p1, p2, pi, pSum
	for j in prange(M):
		for i in range(N):
			pi = P[j,i]
			p0 = L[j,2*i]*(1.0 - pi)*(1.0 - pi)
			p1 = L[j,2*i+1]*2.0*pi*(1.0 - pi)
			p2 = (1.0 - L[j,2*i] - L[j,2*i+1])*pi*pi
			pSum = 1.0/(p0 + p1 + p2)

			# Fill genotype posterior array
			G[j,3*i] = max(1e-10, p0*pSum)
			G[j,3*i+1] = max(1e-10, p1*pSum)
			G[j,3*i+2] = max(1e-10, p2*pSum)

# Genotype posteriors based on individal allele frequencies (inbreeding)
cpdef void postInbreed(const float[:,::1] L, const float[:,::1] P, float[:,::1] G, \
		const double[::1] F) noexcept nogil:
	cdef:
		size_t M = P.shape[0]
		size_t N = P.shape[1]
		size_t i, j
		double p0, p1, p2, pi, pSum
	for j in prange(M):
		for i in range(N):
			pi = P[j,i]
			p0 = L[j,2*i]*((1.0 - pi)*(1.0 - pi) + pi*(1.0 - pi)*F[i])
			p1 = L[j,2*i+1]*2.0*pi*(1.0 - pi)
			p2 = (1.0 - L[j,2*i] - L[j,2*i+1])*(pi*pi + pi*(1.0 - pi)*F[i])
			pSum = 1.0/(p0 + p1 + p2)

			# Fill genotype posterior array
			G[j,3*i] = max(1e-10, p0*pSum)
			G[j,3*i+1] = max(1e-10, p1*pSum)
			G[j,3*i+2] = max(1e-10, p2*pSum)

# Genotype calling
cpdef void geno(const float[:,::1] L, const float[:,::1] P, signed char[:,::1] G, \
		const double delta) noexcept nogil:
	cdef:
		size_t M = P.shape[0]
		size_t N = P.shape[1]
		size_t i, j
		double p0, p1, p2, pi, pSum
	for j in prange(M):
		for i in range(N):
			pi = P[j,i]
			p0 = L[j,2*i]*(1.0 - pi)*(1.0 - pi)
			p1 = L[j,2*i+1]*2.0*pi*(1.0 - pi)
			p2 = (1.0 - L[j,2*i] - L[j,2*i+1])*pi*pi
			pSum = 1.0/(p0 + p1 + p2)

			# Call posterior maximum
			if (p0 > p1) & (p0 > p2):
				if (p0*pSum > delta):
					G[j,i] = 0
				else:
					G[j,i] = -9
			elif (p1 > p2):
				if (p1*pSum > delta):
					G[j,i] = 1
				else:
					G[j,i] = -9
			else:
				if (p2*pSum > delta):
					G[j,i] = 2
				else:
					G[j,i] = -9

# Genotype calling (inbreeding)
cpdef void genoInbreed(const float[:,::1] L, const float[:,::1] P, const double[::1] F, \
		signed char[:,::1] G, const double delta) noexcept nogil:
	cdef:
		size_t M = P.shape[0]
		size_t N = P.shape[1]
		size_t i, j
		double p0, p1, p2, pi, pSum
	for j in prange(M):
		for i in range(N):
			pi = P[j,i]
			p0 = L[j,2*i]*((1.0 - pi)*(1.0 - pi) + pi*(1.0 - pi)*F[i])
			p1 = L[j,2*i+1]*2.0*pi*(1.0 - pi)
			p2 = (1.0 - L[j,2*i] - L[j,2*i+1])*(pi*pi + pi*(1.0 - pi)*F[i])
			pSum = 1.0/(p0 + p1 + p2)

			# Call maximum posterior
			if (p0 > p1) & (p0 > p2):
				if (p0*pSum > delta):
					G[j,i] = 0
				else:
					G[j,i] = -9
			elif (p1 > p2):
				if (p1*pSum > delta):
					G[j,i] = 1
				else:
					G[j,i] = -9
			else:
				if (p2*pSum > delta):
					G[j,i] = 2
				else:
					G[j,i] = -9

# Create fake frequencies
cpdef void freqs(float[:,::1] P, const double[::1] f) noexcept nogil:
	cdef:
		size_t M = P.shape[0]
		size_t N = P.shape[1]
		size_t i, j
		double fj
	for j in prange(M):
		fj = f[j]
		for i in range(N):
			P[j,i] = fj
