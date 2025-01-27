# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
cimport numpy as np
cimport openmp as omp
from cython.parallel import prange, parallel
from libc.math cimport sqrt
from libc.stdlib cimport calloc, free

##### Cython functions for covariance.py ######
# Update posterior expectations (Fumagalli method)
cpdef void updateNormal(const float[:,::1] L, float[:,::1] E, const double[::1] f) \
		noexcept nogil:
	cdef:
		size_t M = L.shape[0]
		size_t N = L.shape[1]//2
		size_t i, j
		double fj, p0, p1, p2
	for j in prange(M):
		fj = f[j]
		for i in range(N):
			# Update dosage
			p0 = L[j,2*i]*(1.0 - fj)*(1.0 - fj)
			p1 = L[j,2*i+1]*2.0*fj*(1.0 - fj)
			p2 = (1.0 - L[j,2*i] - L[j,2*i+1])*fj*fj
			E[j,i] = (p1 + 2.0*p2)/(p0 + p1 + p2) - 2.0*fj

# Update posterior expectations (PCAngsd method)
cpdef void updatePCAngsd(const float[:,::1] L, float[:,::1] P, float[:,::1] E, \
		const double[::1] f) noexcept nogil:
	cdef:
		size_t M = L.shape[0]
		size_t N = L.shape[1]//2
		size_t i, j
		double fj, p0, p1, p2, pi
	for j in prange(M):
		fj = f[j]
		for i in range(N):
			pi = min(max((P[j,i] + 2.0*fj)*0.5, 1e-4), 1.0-(1e-4))

			# Center dosage
			p0 = L[j,2*i]*(1.0 - pi)*(1.0 - pi)
			p1 = L[j,2*i+1]*2.0*pi*(1.0 - pi)
			p2 = (1.0 - L[j,2*i] - L[j,2*i+1])*pi*pi
			E[j,i] = (p1 + 2.0*p2)/(p0 + p1 + p2) - 2.0*fj
			P[j,i] = pi

# Standardize posterior expectations (Fumagalli method)
cpdef void covNormal(const float[:,::1] L, float[:,::1] E, const double[::1] f, \
		double[::1] dCov) noexcept nogil:
	cdef:
		size_t M = L.shape[0]
		size_t N = L.shape[1]//2
		size_t i, j, k
		double dj, fj, p0, p1, p2, pSum, tmp
		double* dPrivate
		omp.omp_lock_t mutex
	omp.omp_init_lock(&mutex)
	with nogil, parallel():
		dPrivate = <double*>calloc(N, sizeof(double))
		for j in prange(M):
			fj = f[j]
			dj = 1.0/sqrt(2.0*fj*(1.0 - fj))
			for i in range(N):
				# Standardize dosage
				p0 = L[j,2*i]*(1.0 - fj)*(1.0 - fj)
				p1 = L[j,2*i+1]*2.0*fj*(1.0 - fj)
				p2 = (1.0 - L[j,2*i] - L[j,2*i+1])*fj*fj
				pSum = p0 + p1 + p2
				E[j,i] = ((p1 + 2.0*p2)/pSum - 2.0*fj)*dj

				# Estimate diagonal
				tmp = (-2.0*fj)*(-2.0*fj)*(p0/pSum)
				tmp = tmp + (1.0 - 2.0*fj)*(1.0 - 2.0*fj)*(p1/pSum)
				tmp = tmp + (2.0 - 2.0*fj)*(2.0 - 2.0*fj)*(p2/pSum)
				dPrivate[i] += tmp/(2.0*fj*(1.0 - fj))
		
		# omp critical
		omp.omp_set_lock(&mutex)
		for k in range(N):
			dCov[k] += dPrivate[k]
		omp.omp_unset_lock(&mutex)
		free(dPrivate)
	omp.omp_destroy_lock(&mutex)

# Standardize posterior expectations (PCAngsd method)
cpdef void covPCAngsd(const float[:,::1] L, float[:,::1] P, float[:,::1] E, \
		const double[::1] f, double[::1] dCov) noexcept nogil:
	cdef:
		size_t M = L.shape[0]
		size_t N = L.shape[1]//2
		size_t i, j, k
		double dj, fj, p0, p1, p2, pi, pSum, tmp
		double* dPrivate
		omp.omp_lock_t mutex
	omp.omp_init_lock(&mutex)
	with nogil, parallel():
		dPrivate = <double*>calloc(N, sizeof(double))
		for j in prange(M):
			fj = f[j]
			dj = 1.0/sqrt(2.0*fj*(1.0 - fj))
			for i in range(N):
				pi = min(max((P[j,i] + 2.0*fj)*0.5, 1e-4), 1.0-(1e-4))

				# Standardize dosage
				p0 = L[j,2*i]*(1.0 - pi)*(1.0 - pi)
				p1 = L[j,2*i+1]*2.0*pi*(1.0 - pi)
				p2 = (1.0 - L[j,2*i] - L[j,2*i+1])*pi*pi
				pSum = p0 + p1 + p2
				E[j,i] = ((p1 + 2.0*p2)/pSum - 2.0*fj)*dj
				P[j,i] = pi

				# Estimate diagonal
				tmp = (-2.0*fj)*(-2.0*fj)*(p0/pSum)
				tmp = tmp + (1.0 - 2.0*fj)*(1.0 - 2.0*fj)*(p1/pSum)
				tmp = tmp + (2.0 - 2.0*fj)*(2.0 - 2.0*fj)*(p2/pSum)
				dPrivate[i] += tmp/(2.0*fj*(1.0 - fj))
		
		# omp critical
		omp.omp_set_lock(&mutex)
		for k in range(N):
			dCov[k] += dPrivate[k]
		omp.omp_unset_lock(&mutex)
		free(dPrivate)
	omp.omp_destroy_lock(&mutex)

# Standardize posterior expectations for selection
cpdef void updateSelection(const float[:,::1] L, const float[:,::1] P, float[:,::1] E, \
		const double[::1] f) noexcept nogil:
	cdef:
		size_t M = L.shape[0]
		size_t N = L.shape[1]//2
		size_t i, j
		double dj, fj, p0, p1, p2, pi
	for j in prange(M):
		fj = f[j]
		dj = 1.0/sqrt(2.0*fj*(1.0 - fj))
		for i in range(N):
			pi = P[j,i]
			p0 = L[j,2*i]*(1.0 - pi)*(1.0 - pi)
			p1 = L[j,2*i+1]*2.0*pi*(1.0 - pi)
			p2 = (1.0 - L[j,2*i] - L[j,2*i+1])*pi*pi
			E[j,i] = ((p1 + 2.0*p2)/(p0 + p1 + p2) - 2.0*fj)*dj

# Update dosages for saving
cpdef void updateDosages(const float[:,::1] L, const float[:,::1] P, float[:,::1] E) \
		noexcept nogil:
	cdef:
		size_t M = L.shape[0]
		size_t N = L.shape[1]//2
		size_t i, j
		double p0, p1, p2, pi
	for j in prange(M):
		for i in range(N):
			pi = P[j,i]
			p0 = L[j,2*i]*(1.0 - pi)*(1.0 - pi)
			p1 = L[j,2*i+1]*2.0*pi*(1.0 - pi)
			p2 = (1.0 - L[j,2*i] - L[j,2*i+1])*pi*pi
			E[j,i] = (p1 + 2.0*p2)/(p0 + p1 + p2)
