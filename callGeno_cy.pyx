import numpy as np
cimport numpy as np
from cython.parallel import prange
from cython import boundscheck, wraparound

##### Cython functions for callGeno.py ######

# Without inbreeding
@boundscheck(False)
@wraparound(False)
cpdef geno(float[:,::1] L, float[:,::1] Pi, float delta, signed char[:,::1] G, int t):
	cdef int n = Pi.shape[0]
	cdef int m = Pi.shape[1]
	cdef int i, j
	cdef float p0, p1, p2, pSum
	with nogil:
		for i in prange(n, num_threads=t):
			for j in range(m):
				p0 = L[3*i,j]*(1 - Pi[i,j])*(1 - Pi[i,j])
				p1 = L[3*i+1,j]*2*Pi[i,j]*(1 - Pi[i,j])
				p2 = L[3*i+2,j]*Pi[i,j]*Pi[i,j]
				pSum = p0 + p1 + p2

				# Call posterior maximum
				if (p0 > p1) & (p0 > p2):
					if (p0/pSum > delta):
						G[i,j] = 0
					else:
						G[i,j] = -9
				elif (p1 > p2):
					if (p1/pSum > delta):
						G[i,j] = 1
					else:
						G[i,j] = -9
				else:
					if (p2/pSum > delta):
						G[i,j] = 2
					else:
						G[i,j] = -9

# With inbreeding
@boundscheck(False)
@wraparound(False)
cpdef genoInbreed(float[:,::1] L, float[:,::1] Pi, float[::1] F, float delta, signed char[:,::1] G, int t):
	cdef int n = Pi.shape[0]
	cdef int m = Pi.shape[1]
	cdef int i, j
	cdef float p0, p1, p2, pSum
	with nogil:
		for i in prange(n, num_threads=t):
			for j in range(m):
				p0 = L[3*i,j]*((1 - Pi[i,j])*(1 - Pi[i,j]) + Pi[i,j]*(1 - Pi[i,j])*F[i])
				p1 = L[3*i+1,j]*(2*Pi[i,j]*(1 - Pi[i,j])*(1 - F[i]))
				p2 = L[3*i+2,j]*(Pi[i,j]*Pi[i,j] + Pi[i,j]*(1 - Pi[i,j])*F[i])
				pSum = p0 + p1 + p2

				# Call posterior maximum
				if (p0 > p1) & (p0 > p2):
					if (p0/pSum > delta):
						G[i,j] = 0
					else:
						G[i,j] = -9
				elif (p1 > p2):
					if (p1/pSum > delta):
						G[i,j] = 1
					else:
						G[i,j] = -9
				else:
					if (p2/pSum > delta):
						G[i,j] = 2
					else:
						G[i,j] = -9