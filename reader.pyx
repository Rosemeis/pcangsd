import os
import numpy as np
cimport numpy as np
from cython.parallel import prange
from cython import boundscheck, wraparound
from libcpp.vector cimport vector
from libc.stdio cimport fopen, fclose, FILE, fread

DTYPE = np.float32
ctypedef np.float32_t DTYPE_t

# Read file
@boundscheck(False)
@wraparound(False)
cpdef readBeagle(str beagle):
	cdef int c = 0
	cdef int n = 0
	cdef int i, m
	cdef list pyList
	cdef vector[vector[float]] L
	cdef vector[float] L_ind
	with os.popen("zcat " + beagle) as f:
		for line in f:
			if c == 0:
				n = len(line.split("\t"))-3
				c += 1
				continue
			pyList = line.split("\t")
			for i in range(3, n+3):
				L_ind.push_back(float(pyList[i]))
			L.push_back(L_ind)
			L_ind.clear()
	m = L.size()
	cdef np.ndarray[DTYPE_t, ndim=2, mode='fortran'] L_np = np.empty((m, n), dtype=DTYPE, order='F')
	cdef float *L_ptr
	for i in range(m):
		L_ptr = &L[i][0]
		L_np[i] = np.asarray(<float[:n]> L_ptr)
	return L_np

# Read file (remove related individuals)
@boundscheck(False)
@wraparound(False)
cpdef readBeagleUnrelated(beagle, relBool):
	cdef int c = 0
	cdef int n = 0
	cdef int nU = 0
	cdef int i, m
	cdef list pyList
	cdef vector[vector[float]] L
	cdef vector[float] L_ind
	cdef np.uint8_t[:] relBoolView = np.frombuffer(relBool, dtype=np.uint8)
	with os.popen("zcat " + beagle) as f:
		for line in f:
			if c == 0:
				n = len(line.split("\t"))-3
				nU = np.sum(relBool)
				c += 1
				continue
			pyList = line.split("\t")
			for i in range(3, n+3):
				if relBoolView[i-3] == True:
					L_ind.push_back(float(pyList[i]))
			L.push_back(L_ind)
			L_ind.clear()
	m = L.size()
	cdef np.ndarray[DTYPE_t, ndim=2, mode='fortran'] L_np = np.empty((m, nU), dtype=DTYPE, order='F')
	cdef float *L_ptr
	for i in range(m):
		L_ptr = &L[i][0]
		L_np[i] = np.asarray(<float[:nU]> L_ptr)
	return L_np

# Read .bed file
@boundscheck(False)
@wraparound(False)
cpdef readBed(str bedfile, signed char[:,::1] G, int n, int m):
	cdef signed char[4] recode = [0, -9, 1, 2] # PCAngsd format
	cdef unsigned char mask = 3
	cdef unsigned char byte, code
	cdef unsigned char start[3]
	cdef int bytepart = 0, i = 0, j = 0
	cdef FILE *bed = fopen(bedfile.encode('utf-8'), "r")
	fread(start, 1, 3, bed) # Read first three bytes - 0x6c, 0x1b, and 0x01

	while True:
		if bytepart == 0: # Read byte
			fread(&byte, 1, 1, bed)
			bytepart = 4
		code = byte & mask
		G[i,j] = recode[code]
		byte = byte >> 2
		bytepart -= 1
		i += 1
		if i == n:
			i = 0
			bytepart = 0
			j += 1
			if j == m:
				break
	fclose(bed)

# PLINK converter to genotype likelihoods
@boundscheck(False)
@wraparound(False)
cpdef convertBed(float[:,::1] L, signed char[:,::1] G, float e, int t):
	cdef int n = G.shape[0]
	cdef int m = G.shape[1]
	cdef int i, j
	with nogil:
		for i in prange(n, num_threads=t):
			for j in range(m):
				if G[i,j] == -9: # Missing site
					L[3*i,j] = 0.333333
					L[3*i+1,j] = 0.333333
					L[3*i+2,j] = 0.333333
				elif G[i,j] == 0:
					L[3*i,j] = e*e
					L[3*i+1,j] = 2*(1 - e)*e
					L[3*i+2,j] = (1 - e)*(1 - e)
				elif G[i,j] == 1:
					L[3*i,j] = (1 - e)*e
					L[3*i+1,j] = (1 - e)*(1 - e) + e*e
					L[3*i+2,j] = (1 - e)*e
				elif G[i,j] == 2:
					L[3*i,j] = (1 - e)*(1 - e)
					L[3*i+1,j] = 2*(1 - e)*e
					L[3*i+2,j] = e*e