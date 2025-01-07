# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
import os
import numpy as np
cimport numpy as np
from cython.parallel import prange
from libcpp.vector cimport vector
#from libcpp.iostrem cimport cout scientific
from libc.string cimport strtok, strdup
from libc.stdlib cimport atof
from libc.stdio cimport FILE, fclose, fopen, fprintf

DTYPE = np.float32
ctypedef np.float32_t DTYPE_t

# Read Beagle text format
cpdef np.ndarray[DTYPE_t, ndim=2] readBeagle(str beagle):
	cdef:
		int c = 0
		int i, j, m, n
		bytes line_bytes
		str line_str
		char* line
		char* token
		char* delims = "\t \n"
		float *L_ptr
		vector[vector[float]] L
		vector[float] L_ind
		np.ndarray[DTYPE_t, ndim=2] L_np
	with os.popen("zcat " + beagle) as f:
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
				if (i + 1) % 3 == 0:
					token = strtok(NULL, delims)
				else:
					L_ind.push_back(atof(strtok(NULL, delims)))
			L.push_back(L_ind)
			L_ind.clear()
	m = L.size() # Number of sites
	L_np = np.empty((m, (n//3)*2), dtype=DTYPE)
	for j in range(m):
		L_ptr = &L[j][0]
		L_np[j] = np.asarray(<float[:((n//3)*2)]> L_ptr)
	return L_np

# Read Beagle text format and filtering sites
cpdef np.ndarray[DTYPE_t, ndim=2] readBeagleFilterSites(str beagle, \
		unsigned char[::1] F):
	cdef:
		int c = 0
		int j = 0
		int i, m, n
		bytes line_bytes
		str line_str
		char* line
		char* token
		char* delims = "\t \n"
		float *L_ptr
		vector[vector[float]] L
		vector[float] L_ind
		np.ndarray[DTYPE_t, ndim=2] L_np
	with os.popen("zcat " + beagle) as f:
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
			if F[j] == 1:
				line_bytes = str.encode(line_str)
				line = line_bytes
				token = strtok(line, delims)
				token = strtok(NULL, delims)
				token = strtok(NULL, delims)
				for i in range(n):
					if (i + 1) % 3 == 0:
						token = strtok(NULL, delims)
					else:
						L_ind.push_back(atof(strtok(NULL, delims)))
				L.push_back(L_ind)
				L_ind.clear()
			j += 1
	m = L.size() # Number of sites
	L_np = np.empty((m, (n//3)*2), dtype=DTYPE)
	for j in range(m):
		L_ptr = &L[j][0]
		L_np[j] = np.asarray(<float[:((n//3)*2)]> L_ptr)
	return L_np

# Read Beagle text format and filtering individuals
cpdef np.ndarray[DTYPE_t, ndim=2] readBeagleFilterInd(str beagle, \
		unsigned char[::1] F, int N):
	cdef:
		int c = 0
		int i, j, m, n
		bytes line_bytes
		str line_str
		char* line
		char* token
		char* delims = "\t \n"
		float *L_ptr
		vector[vector[float]] L
		vector[float] L_ind
		np.ndarray[DTYPE_t, ndim=2] L_np
	with os.popen("zcat " + beagle) as f:
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
				if (i + 1) % 3 == 0:
					token = strtok(NULL, delims)
				elif F[i] == 0:
					token = strtok(NULL, delims)
				else:
					L_ind.push_back(atof(strtok(NULL, delims)))
			L.push_back(L_ind)
			L_ind.clear()
	m = L.size() # Number of sites
	L_np = np.empty((m, (N//3)*2), dtype=DTYPE)
	for j in range(m):
		L_ptr = &L[j][0]
		L_np[j] = np.asarray(<float[:((N//3)*2)]> L_ptr)
	return L_np

# Read Beagle text format and filtering sites and individuals
cpdef np.ndarray[DTYPE_t, ndim=2] readBeagleFilter(str beagle, \
		unsigned char[::1] Fj, unsigned char[::1] Fi, int N):
	cdef:
		int c = 0, j = 0
		int i, m, n
		bytes line_bytes
		str line_str
		char* line
		char* token
		char* delims = "\t \n"
		float *L_ptr
		vector[vector[float]] L
		vector[float] L_ind
		np.ndarray[DTYPE_t, ndim=2] L_np
	with os.popen("zcat " + beagle) as f:
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
					if (i + 1) % 3 == 0:
						token = strtok(NULL, delims)
					elif Fi[i] == 0:
						token = strtok(NULL, delims)
					else:
						L_ind.push_back(atof(strtok(NULL, delims)))
				L.push_back(L_ind)
				L_ind.clear()
			j += 1
	m = L.size() # Number of sites
	L_np = np.empty((m, (N//3)*2), dtype=DTYPE)
	for j in range(m):
		L_ptr = &L[j][0]
		L_np[j] = np.asarray(<float[:((N//3)*2)]> L_ptr)
	return L_np

# Convert PLINK bed format to Beagle format
cpdef void convertBed(float[:,::1] L, unsigned char[:,::1] G, int G_len, float e, \
		int m, int n, int t) noexcept nogil:
	cdef:
		int i, j, b, bytepart
		unsigned char byte, code
		unsigned char mask = 3
		unsigned char[4] recode = [0, 9, 1, 2]
	for j in prange(m, num_threads=t):
		i = 0
		for b in range(G_len):
			byte = G[j,b]
			for bytepart in range(4):
				code = recode[byte & mask]
				if code == 0:
					L[j,2*i+0] = e*e
					L[j,2*i+1] = 2*e*(1 - e)
				elif code == 1:
					L[j,2*i+0] = (1 - e)*e
					L[j,2*i+1] = (1 - e)*(1 - e) + e*e
				elif code == 2:
					L[j,2*i+0] = (1 - e)*(1 - e)
					L[j,2*i+1] = 2*e*(1 - e)
				else:
					L[j,2*i+0] = 0.333333
					L[j,2*i+1] = 0.333333
				byte = byte >> 2
				i = i + 1
				if i == n:
					break

# Array filtering


# Eric is going to try to put a little C-ish function here
# for writing out the genotype posteriors
cpdef void writeBeagle(float[:,::1] Po, str beagle):
    cdef int m = Po.shape[0]
    cdef int n3 = Po.shape[1]
    cdef int c = 0
    cdef int i, s
    #cdef FILE *outf
    cdef FILE *pipe

    #pipe = popen(str.encode("gzip - > " + beagle), "wb");
    outf = fopen(str.encode(beagle), "wb")
    for s in range(m):
        for i in range(n3):
            if i == 0:
                fprintf(outf, "%1.4e", Po[s, i])
            else:
                fprintf(outf, "\t%1.4e", Po[s, i])
            if i == n3-1:
                fprintf(outf, "\n")
    fclose(outf)


cpdef void filterArrays(float[:,::1] L, double[::1] f, unsigned char[::1] mask) \
		noexcept nogil:
	cdef:
		int m = L.shape[0]
		int n = L.shape[1]
		int c = 0
		int i, j
	for j in range(m):
		if mask[j] == 1:
			for i in range(n):
				L[c,i] = L[j,i] # Genotype likelihoods
			f[c] = f[j] # Allele frequency
			c += 1

