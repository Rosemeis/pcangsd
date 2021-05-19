import os
import numpy as np
cimport numpy as np
from cython import boundscheck, wraparound
from cython.parallel import prange
from libcpp.vector cimport vector
from libc.string cimport strtok, strdup
from libc.stdlib cimport atof

DTYPE = np.float32
ctypedef np.float32_t DTYPE_t

# Read Beagle text format
@boundscheck(False)
@wraparound(False)
cpdef np.ndarray[DTYPE_t, ndim=2] readBeagle(str beagle):
    cdef int c = 0
    cdef int i, m, n, s
    cdef bytes line_bytes
    cdef str line_str
    cdef char* line
    cdef char* token
    cdef char* delims = "\t \n"
    cdef vector[vector[float]] L
    cdef vector[float] L_ind
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
    cdef np.ndarray[DTYPE_t, ndim=2] L_np = np.empty((m, n//3*2), dtype=DTYPE)
    cdef float *L_ptr
    for s in range(m):
        L_ptr = &L[s][0]
        L_np[s] = np.asarray(<float[:(n//3*2)]> L_ptr)
    return L_np

# Read Beagle text format and filtering individuals
@boundscheck(False)
@wraparound(False)
cpdef np.ndarray[DTYPE_t, ndim=2] readBeagleFilter(str beagle, \
                                                    unsigned char[::1] F, int N):
    cdef int c = 0
    cdef int i, m, n, s
    cdef bytes line_bytes
    cdef str line_str
    cdef char* line
    cdef char* token
    cdef char* delims = "\t \n"
    cdef vector[vector[float]] L
    cdef vector[float] L_ind
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
                if (F[i] == 1) or ((i + 1) % 3 != 0):
                    L_ind.push_back(atof(strtok(NULL, delims)))
                else:
                    token = strtok(NULL, delims)
            L.push_back(L_ind)
            L_ind.clear()
    m = L.size() # Number of sites
    cdef np.ndarray[DTYPE_t, ndim=2] L_np = np.empty((m, N//3*2), dtype=DTYPE)
    cdef float *L_ptr
    for s in range(m):
        L_ptr = &L[s][0]
        L_np[s] = np.asarray(<float[:(N//3*2)]> L_ptr)
    return L_np

# Convert PLINK bed format to Beagle format
@boundscheck(False)
@wraparound(False)
cpdef convertBed(float[:,::1] L, unsigned char[:,::1] G, int G_len, float e, \
                    int m, int n, int t):
    cdef signed char[4] recode = [0, 9, 1, 2]
    cdef unsigned char mask = 3
    cdef unsigned char byte, code
    cdef int i, s, b, bytepart
    with nogil:
        for s in prange(m, num_threads=t):
            i = 0
            for b in range(G_len):
                byte = G[s,b]
                for bytepart in range(4):
                    code = recode[byte & mask]
                    if code == 0:
                        L[s,3*i+0] = e*e
                        L[s,3*i+1] = 2*e*(1 - e)
                    elif code == 1:
                        L[s,3*i+0] = (1 - e)*e
                        L[s,3*i+1] = (1 - e)*(1 - e) + e*e
                    elif code == 2:
                        L[s,3*i+0] = (1 - e)*(1 - e)
                        L[s,3*i+1] = 2*e*(1 - e)
                    else:
                        L[s,3*i+0] = 0.333333
                        L[s,3*i+1] = 0.333333
                    byte = byte >> 2
                    i = i + 1
                    if i == n:
                        break

# Array filtering
@boundscheck(False)
@wraparound(False)
cpdef filterArrays(float[:,::1] L, float[::1] f, unsigned char[::1] mask):
    cdef int m = L.shape[0]
    cdef int n = L.shape[1]
    cdef int c = 0
    cdef int i, s
    for s in range(m):
        if mask[s] == 1:
            for i in range(n):
                L[c,i] = L[s,i] # Genotype likelihoods
            f[c] = f[s] # Allele frequency
            c += 1
