# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
cimport numpy as np
cimport openmp as omp
from cython.parallel import prange, parallel
from libc.math cimport log
from libc.stdlib cimport calloc, free

##### Cython functions for inbreed.py #####
# Per-site
cpdef void inbreedSites_update(const float[:,::1] L, const float[:,::1] P, \
		double[::1] F) noexcept nogil:
	cdef:
		size_t M = P.shape[0]
		size_t N = P.shape[1]
		size_t i, j
		double expH, obsH, p0, p1, p2, pSum, Fj, Pi
	for j in prange(M):
		Fj = F[j]
		expH = 0.0
		obsH = 0.0
		for i in range(N):
			Pi = P[j,i]
			p0 = max(1e-4, (1.0 - Pi)*(1.0 - Pi) + Pi*(1.0 - Pi)*Fj)
			p1 = max(1e-4, 2.0*Pi*(1.0 - Pi)*(1.0 - Fj))
			p2 = max(1e-4, Pi*Pi + Pi*(1.0 - Pi)*Fj)
			pSum = 1.0/(p0 + p1 + p2)

			# Readjust distribution and estimate posterior
			p0 = p0*pSum*L[j,2*i]
			p1 = p1*pSum*L[j,2*i+1]
			p2 = p2*pSum*(1.0 - L[j,2*i] - L[j,2*i+1])

			# Count heterozygotes
			obsH = obsH + p1/(p0 + p1 + p2)
			expH = expH + 2.0*Pi*(1.0 - Pi)

		# Update inbreeding coefficient
		F[j] = min(max(-1.0, 1.0 - (obsH/expH)), 1.0)

# Per-site accelerated update
cpdef void inbreedSites_accel(const float[:,::1] L, const float[:,::1] P, \
		const double[::1] F, double[::1] F_new) noexcept nogil:
	cdef:
		size_t M = P.shape[0]
		size_t N = P.shape[1]
		size_t i, j
		double expH, obsH, p0, p1, p2, pSum, Fj, Pi
	for j in prange(M):
		Fj = F[j]
		expH = 0.0
		obsH = 0.0
		for i in range(N):
			Pi = P[j,i]
			p0 = max(1e-4, (1.0 - Pi)*(1.0 - Pi) + Pi*(1.0 - Pi)*Fj)
			p1 = max(1e-4, 2.0*Pi*(1.0 - Pi)*(1.0 - Fj))
			p2 = max(1e-4, Pi*Pi + Pi*(1.0 - Pi)*Fj)
			pSum = 1.0/(p0 + p1 + p2)

			# Readjust distribution and estimate posterior
			p0 = p0*pSum*L[j,2*i]
			p1 = p1*pSum*L[j,2*i+1]
			p2 = p2*pSum*(1.0 - L[j,2*i] - L[j,2*i+1])

			# Count heterozygotes
			obsH = obsH + p1/(p0 + p1 + p2)
			expH = expH + 2.0*Pi*(1.0 - Pi)

		# Update inbreeding coefficient
		F_new[j] = min(max(-1.0, 1.0 - (obsH/expH)), 1.0)

# Log-likelihoods
cpdef void loglike(const float[:,::1] L, const float[:,::1] P, const double[::1] F, \
		double[::1] T) noexcept nogil:
	cdef:
		size_t M = P.shape[0]
		size_t N = P.shape[1]
		size_t i, j
		double l0, l1, l2, logA, logN, p0, p1, p2, pSum, Fj, Pi
	for j in prange(M):
		Fj = F[j]
		logA = 0.0
		logN = 0.0
		for i in range(N):
			Pi = P[j,i]

			# Alternative model
			p0 = max(1e-4, (1.0 - Pi)*(1.0 - Pi) + Pi*(1.0 - Pi)*Fj)
			p1 = max(1e-4, 2.0*Pi*(1.0 - Pi)*(1.0 - Fj))
			p2 = max(1e-4, Pi*Pi + Pi*(1.0 - Pi)*Fj)
			pSum = 1.0/(p0 + p1 + p2)

			# Readjust distribution and posterior
			p0 = p0*pSum*L[j,2*i]
			p1 = p1*pSum*L[j,2*i+1]
			p2 = p2*pSum*(1.0 - L[j,2*i] - L[j,2*i+1])
			logA = logA + log(p0 + p1 + p2)

			# Null model
			l0 = L[j,2*i]*(1.0 - Pi)*(1.0 - Pi)
			l1 = L[j,2*i+1]*2.0*Pi*(1.0 - Pi)
			l2 = (1.0 - L[j,2*i] - L[j,2*i+1])*Pi*Pi
			logN = logN + log(l0 + l1 + l2)
		T[j] = 2.0*(logA - logN)

# Per-sample
cpdef void inbreedSamples_update(const float[:,::1] L, const float[:,::1] P, \
		double[::1] F, double[::1] Ftmp, double[::1] Etmp) noexcept nogil:
	cdef:
		size_t M = P.shape[0]
		size_t N = P.shape[1]
		size_t h, i, j, k, l
		double p0, p1, p2, pSum, Pi
		double* obsH
		double* expH
		omp.omp_lock_t mutex
	omp.omp_init_lock(&mutex)
	for h in range(N):
		Ftmp[h] = 0.0
		Etmp[h] = 0.0
	with nogil, parallel():
		obsH = <double*>calloc(N, sizeof(double))
		expH = <double*>calloc(N, sizeof(double))
		for j in prange(M):
			for i in range(N):
				Pi = P[j,i]
				p0 = max(1e-4, (1.0 - Pi)*(1.0 - Pi) + Pi*(1.0 - Pi)*F[i])
				p1 = max(1e-4, 2.0*Pi*(1.0 - Pi)*(1.0 - F[i]))
				p2 = max(1e-4, Pi*Pi + Pi*(1.0 - Pi)*F[i])
				pSum = 1.0/(p0 + p1 + p2)

				# Readjust distribution and estimate posterior
				p0 = p0*pSum*L[j,2*i]
				p1 = p1*pSum*L[j,2*i+1]
				p2 = p2*pSum*(1.0 - L[j,2*i] - L[j,2*i+1])

				# Count heterozygotes
				obsH[i] += p1/(p0 + p1 + p2)
				expH[i] += 2.0*Pi*(1.0 - Pi)

		# omp critical
		omp.omp_set_lock(&mutex)
		for k in range(N):
			Ftmp[k] += obsH[k]
			Etmp[k] += expH[k]
		omp.omp_unset_lock(&mutex)
		free(obsH)
		free(expH)
	omp.omp_destroy_lock(&mutex)

	# Truncate inbreeding coefficients 
	for l in range(N):
		F[l] = min(max(-1.0, 1.0 - (Ftmp[l]/Etmp[l])), 1.0)

# Per-sample accelerated update
cpdef void inbreedSamples_accel(const float[:,::1] L, const float[:,::1] P, \
		const double[::1] F, double[::1] F_new, double[::1] Ftmp, double[::1] Etmp) \
		noexcept nogil:
	cdef:
		size_t M = P.shape[0]
		size_t N = P.shape[1]
		size_t h, i, j, k, l
		double p0, p1, p2, pSum, Pi
		double* obsH
		double* expH
		omp.omp_lock_t mutex
	omp.omp_init_lock(&mutex)
	for h in range(N):
		Ftmp[h] = 0.0
		Etmp[h] = 0.0
	with nogil, parallel():
		obsH = <double*>calloc(N, sizeof(double))
		expH = <double*>calloc(N, sizeof(double))
		for j in prange(M):
			for i in range(N):
				Pi = P[j,i]
				p0 = max(1e-4, (1.0 - Pi)*(1.0 - Pi) + Pi*(1.0 - Pi)*F[i])
				p1 = max(1e-4, 2.0*Pi*(1.0 - Pi)*(1.0 - F[i]))
				p2 = max(1e-4, Pi*Pi + Pi*(1.0 - Pi)*F[i])
				pSum = 1.0/(p0 + p1 + p2)

				# Readjust distribution and estimate posterior
				p0 = p0*pSum*L[j,2*i]
				p1 = p1*pSum*L[j,2*i+1]
				p2 = p2*pSum*(1.0 - L[j,2*i] - L[j,2*i+1])

				# Count heterozygotes
				obsH[i] += p1/(p0 + p1 + p2)
				expH[i] += 2.0*Pi*(1.0 - Pi)
		
		# omp critical
		omp.omp_set_lock(&mutex)
		for k in range(N):
			Ftmp[k] += obsH[k]
			Etmp[k] += expH[k]
		omp.omp_unset_lock(&mutex)
		free(obsH)
		free(expH)
	omp.omp_destroy_lock(&mutex)

	# Truncate inbreeding coefficients 
	for l in range(N):
		F_new[l] = min(max(-1.0, 1.0 - (Ftmp[l]/Etmp[l])), 1.0)
