"""
Call genotypes from posterior genotype probabilities using estimated individual allele frequencies as prior.
Can be performed with and without taking inbreeding into account.
"""

__author__ = "Jonas Meisner"

# Import libraries
import numpy as np
import threading
from numba import jit

##### Functions #####
# Genotype calling without inbreeding
@jit("void(f4[:, :], f4[:, :], f8, i8, i8, i1[:, :])", nopython=True, nogil=True, cache=True)
def gProbGeno(likeMatrix, indF, delta, S, N, G):
	m, n = indF.shape # Dimensions
	
	for ind in xrange(S, min(S+N, m)):
		# Estimate posterior probabilities
		for s in xrange(n):
			p0 = likeMatrix[3*ind, s]*(1 - indF[ind, s])*(1 - indF[ind, s])
			p1 = likeMatrix[3*ind+1, s]*2*indF[ind, s]*(1 - indF[ind, s])
			p2 = likeMatrix[3*ind+2, s]*indF[ind, s]*indF[ind, s]
			pSum = p0 + p1 + p2

			if (p0 == p1) & (p0 == p2): # If all equal
				G[ind, s] = -9
				continue

			# Call posterior maximum
			if p0 > p1:
				if p0 > p2: # G = 0
					if p0/pSum > delta:
						G[ind, s] = 0
					else:
						G[ind, s] = -9
				else: # G = 2
					if p2/pSum > delta:
						G[ind, s] = 2
					else:
						G[ind, s] = -9

			else: # G = 1
				if p1/pSum > delta:
					G[ind, s] = 1
				else:
					G[ind, s] = -9

# Genotype calling with inbreeding
@jit("void(f4[:, :], f4[:, :], f4[:], f8, i8, i8, i1[:, :])", nopython=True, nogil=True, cache=True)
def gProbGenoInbreeding(likeMatrix, indF, F, delta, S, N, G):
	m, n = indF.shape # Dimensions
	
	for ind in xrange(S, min(S+N, m)):
		# Estimate posterior probabilities
		for s in xrange(n):
			p0 = likeMatrix[3*ind, s]*((1 - indF[ind, s])*(1 - indF[ind, s]) + indF[ind, s]*(1 - indF[ind, s])*F[ind])
			p1 = likeMatrix[3*ind+1, s]*(2*indF[ind, s]*(1 - indF[ind, s])*(1 - F[ind]))
			p2 = likeMatrix[3*ind+2, s]*(indF[ind, s]*indF[ind, s] + indF[ind, S]*(1 - indF[ind, S])*F[ind])
			pSum = p0 + p1 + p2

			if (p0 == p1) & (p0 == p2): # If all equal
				G[ind, s] = -9
				continue

			# Call posterior maximum
			if p0 > p1:
				if p0 > p2: # G = 0
					if p0/pSum > delta:
						G[ind, s] = 0
					else:
						G[ind, s] = -9
				else: # G = 2
					if p2/pSum > delta:
						G[ind, s] = 2
					else:
						G[ind, s] = -9

			else: # G = 1
				if p1/pSum > delta:
					G[ind, s] = 1
				else:
					G[ind, s] = -9

##### Genotype calling #####
def callGeno(likeMatrix, indF, F=None, delta=0.0, threads=1):
	m, n = indF.shape # Dimensions
	chunk_N = int(np.ceil(float(m)/threads))
	chunks = [i * chunk_N for i in xrange(threads)]

	# Initiate genotype matrix
	G = np.empty((m, n), dtype=np.int8)

	# Call genotypes with highest posterior probabilities
	if type(F) != type(None):
		# Multithreading
		threads = [threading.Thread(target=gProbGenoInbreeding, args=(likeMatrix, indF, F, delta, chunk, chunk_N, G)) for chunk in chunks]
		for thread in threads:
			thread.start()
		for thread in threads:
			thread.join()
	else:
		# Multithreading
		threads = [threading.Thread(target=gProbGeno, args=(likeMatrix, indF, delta, chunk, chunk_N, G)) for chunk in chunks]
		for thread in threads:
			thread.start()
		for thread in threads:
			thread.join()

	return G