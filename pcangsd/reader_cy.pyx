# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
import os
import numpy as np
cimport numpy as np
from cython.parallel import prange
from libc.stdio cimport FILE, fclose, fopen, fprintf
from libc.stdlib cimport atof
from libc.string cimport strtok, strdup
from libcpp.vector cimport vector

DTYPE = np.float32
ctypedef np.float32_t DTYPE_t

# Read Beagle text format
cpdef np.ndarray[DTYPE_t, ndim=2] readBeagle(str beagle):
	cdef:
		size_t c = 0
		size_t i, j, M, N
		bytes line_bytes
		str line_str
		char* line
		char* token
		char* delims = "\t \n"
		float* L_ptr
		vector[vector[float]] L
		vector[float] L_ind
		np.ndarray[DTYPE_t, ndim=2] L_np
	with os.popen("gzip -cd " + beagle) as f:
		# Count number of individuals from first line
		line_bytes = str.encode(f.readline())
		line = line_bytes
		token = strtok(line, delims)
		while token != NULL:
			token = strtok(NULL, delims)
			c += 1
		N = c - 3

		# Add lines to vector
		for line_str in f:
			line_bytes = str.encode(line_str)
			line = line_bytes
			token = strtok(line, delims)
			token = strtok(NULL, delims)
			token = strtok(NULL, delims)
			for i in range(N):
				if (i+1) % 3 == 0:
					token = strtok(NULL, delims)
				else:
					L_ind.push_back(atof(strtok(NULL, delims)))
			L.push_back(L_ind)
			L_ind.clear()
	M = L.size() # Number of sites
	L_np = np.empty((M, (N//3)*2), dtype=DTYPE)
	for j in range(M):
		L_ptr = &L[j][0]
		L_np[j] = np.asarray(<float[:((N//3)*2)]> L_ptr)
	return L_np

# Read Beagle text format and filtering sites
cpdef np.ndarray[DTYPE_t, ndim=2] readBeagleFilterSites(str beagle, \
		const unsigned char[::1] F):
	cdef:
		size_t c = 0
		size_t j = 0
		size_t i, M, N
		bytes line_bytes
		str line_str
		char* line
		char* token
		char* delims = "\t \n"
		float* L_ptr
		vector[vector[float]] L
		vector[float] L_ind
		np.ndarray[DTYPE_t, ndim=2] L_np
	with os.popen("gzip -cd " + beagle) as f:
		# Count number of individuals from first line
		line_bytes = str.encode(f.readline())
		line = line_bytes
		token = strtok(line, delims)
		while token != NULL:
			token = strtok(NULL, delims)
			c += 1
		N = c - 3

		# Add lines to vector
		for line_str in f:
			if F[j] == 1:
				line_bytes = str.encode(line_str)
				line = line_bytes
				token = strtok(line, delims)
				token = strtok(NULL, delims)
				token = strtok(NULL, delims)
				for i in range(N):
					if (i+1) % 3 == 0:
						token = strtok(NULL, delims)
					else:
						L_ind.push_back(atof(strtok(NULL, delims)))
				L.push_back(L_ind)
				L_ind.clear()
			j += 1
	M = L.size() # Number of sites
	L_np = np.empty((M, (N//3)*2), dtype=DTYPE)
	for j in range(M):
		L_ptr = &L[j][0]
		L_np[j] = np.asarray(<float[:((N//3)*2)]> L_ptr)
	return L_np

# Read Beagle text format and filtering individuals
cpdef np.ndarray[DTYPE_t, ndim=2] readBeagleFilterInd(str beagle, \
		unsigned char[::1] F, size_t N):
	cdef:
		size_t c = 0
		size_t i, j, n, M
		bytes line_bytes
		str line_str
		char* line
		char* token
		char* delims = "\t \n"
		float* L_ptr
		vector[vector[float]] L
		vector[float] L_ind
		np.ndarray[DTYPE_t, ndim=2] L_np
	with os.popen("gzip -cd " + beagle) as f:
		# Count number of individuals from first line
		line_bytes = str.encode(f.readline())
		line = line_bytes
		token = strtok(line, delims)
		while token != NULL:
			token = strtok(NULL, delims)
			c += 1
		n = c - 3

		# Add lines to vector
		for line_str in f:
			line_bytes = str.encode(line_str)
			line = line_bytes
			token = strtok(line, delims)
			token = strtok(NULL, delims)
			token = strtok(NULL, delims)
			for i in range(n):
				if (i+1) % 3 == 0:
					token = strtok(NULL, delims)
				elif F[i] == 0:
					token = strtok(NULL, delims)
				else:
					L_ind.push_back(atof(strtok(NULL, delims)))
			L.push_back(L_ind)
			L_ind.clear()
	M = L.size() # Number of sites
	L_np = np.empty((M, (N//3)*2), dtype=DTYPE)
	for j in range(M):
		L_ptr = &L[j][0]
		L_np[j] = np.asarray(<float[:((N//3)*2)]> L_ptr)
	return L_np

# Read Beagle text format and filtering sites and individuals
cpdef np.ndarray[DTYPE_t, ndim=2] readBeagleFilter(str beagle, \
		const unsigned char[::1] Fj, const unsigned char[::1] Fi, const size_t N):
	cdef:
		size_t c = 0, j = 0
		size_t i, n, M
		bytes line_bytes
		str line_str
		char* line
		char* token
		char* delims = "\t \n"
		float* L_ptr
		vector[vector[float]] L
		vector[float] L_ind
		np.ndarray[DTYPE_t, ndim=2] L_np
	with os.popen("gzip -cd " + beagle) as f:
		# Count number of individuals from first line
		line_bytes = str.encode(f.readline())
		line = line_bytes
		token = strtok(line, delims)
		while token != NULL:
			token = strtok(NULL, delims)
			c += 1
		n = c - 3

		# Add lines to vector
		for line_str in f:
			if Fj[j] == 1:
				line_bytes = str.encode(line_str)
				line = line_bytes
				token = strtok(line, delims)
				token = strtok(NULL, delims)
				token = strtok(NULL, delims)
				for i in range(n):
					if (i+1) % 3 == 0:
						token = strtok(NULL, delims)
					elif Fi[i] == 0:
						token = strtok(NULL, delims)
					else:
						L_ind.push_back(atof(strtok(NULL, delims)))
				L.push_back(L_ind)
				L_ind.clear()
			j += 1
	M = L.size() # Number of sites
	L_np = np.empty((M, (N//3)*2), dtype=DTYPE)
	for j in range(M):
		L_ptr = &L[j][0]
		L_np[j] = np.asarray(<float[:((N//3)*2)]> L_ptr)
	return L_np

# Convert PLINK bed format to Beagle format
cpdef void convertBed(float[:,::1] L, const unsigned char[:,::1] G, const float e) \
		noexcept nogil:
	cdef:
		size_t M = L.shape[0]
		size_t B = G.shape[1]
		size_t N = L.shape[1]//2
		size_t i, j, b, bytepart
		unsigned char g, byte
		unsigned char mask = 3
		unsigned char[4] recode = [2, 9, 1, 0]
	for j in prange(M):
		i = 0
		for b in range(B):
			byte = G[j,b]
			for bytepart in range(4):
				g = recode[byte & mask]
				if g == 0:
					L[j,2*i] = e*e
					L[j,2*i+1] = 2*e*(1 - e)
				elif g == 1:
					L[j,2*i] = (1 - e)*e
					L[j,2*i+1] = (1 - e)*(1 - e) + e*e
				elif g == 2:
					L[j,2*i] = (1 - e)*(1 - e)
					L[j,2*i+1] = 2*e*(1 - e)
				else:
					L[j,2*i] = 0.333333
					L[j,2*i+1] = 0.333333
				byte = byte >> 2
				i = i + 1
				if i == N:
					break

# Array filtering
cpdef void filterArrays(float[:,::1] L, double[::1] f, const unsigned char[::1] mask) \
		noexcept nogil:
	cdef:
		size_t M = L.shape[0]
		size_t N = L.shape[1]
		size_t c = 0
		size_t i, j
	for j in range(M):
		if mask[j] == 1:
			for i in range(N):
				L[c,i] = L[j,i]
			f[c] = f[j]
			c += 1

# Write genotype posteriors to beagle text-format
cpdef void writeBeagle(const float[:,::1] G, str beagle):
	cdef:
		size_t M = G.shape[0]
		size_t N = G.shape[1]
		size_t i, j
		FILE *f
	f = fopen(str.encode(beagle), "wb")
	for j in range(M):
		for i in range(N):
			if i == 0:
				fprintf(f, "%.6f", G[j,i])
			else:
				fprintf(f, "\t%.6f", G[j,i])
			if i == (N-1):
				fprintf(f, "\n")
	fclose(f)
