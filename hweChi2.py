"""
Test for Hardy-Weinberg Equilibrium in structured populations using genotype likelihoods. 
"""

__author__ = "Jonas Meisner"

# Import libraries
import numpy as np
import threading
from numba import jit


# Inner multithreaded function of HWE test 
@jit("void(f4[:, :], f4[:, :], i8, i8, f8[:])", nopython=True, nogil=True, cache=True)
def hweTest_inner(likeMatrix, indF, S, N, H):
	m, n = indF.shape # Dimensions
	probMatrix = np.empty((3, m)) # Container for posterior probabilities

	for s in xrange(S, min(S+N, n)):
		expH = np.zeros(3)

		for ind in xrange(m):
			probMatrix[0, ind] = likeMatrix[3*ind, s]*(1 - indF[ind, s])*(1 - indF[ind, s])
			probMatrix[1, ind] = likeMatrix[3*ind + 1, s]*2*indF[ind, s]*(1 - indF[ind, s])
			probMatrix[2, ind] = likeMatrix[3*ind + 2, s]*indF[ind, s]*indF[ind, s]
			expH[0] += (1 - indF[ind, s])*(1 - indF[ind, s])
			expH[1] += 2*indF[ind, s]*(1 - indF[ind, s])
			expH[2] += indF[ind, s]*indF[ind, s]
		probMatrix /= np.sum(probMatrix, axis=0)

		for g in xrange(3):
			H[s] += ((np.sum(probMatrix[g, :]) - expH[g])**2)/expH[g]


# Estimate observed and expected counts of genotypes and compute chi2 test statistic
def hweTest(likeMatrix, indF, t=1):
	m, n = indF.shape
	hwe = np.zeros(n)

	# Multithreading parameters
	chunk_N = int(np.ceil(float(n)/t))
	chunks = [i * chunk_N for i in xrange(t)]

	# Multithreading
	threads = [threading.Thread(target=hweTest_inner, args=(likeMatrix, indF, chunk, chunk_N, hwe)) for chunk in chunks]
	for thread in threads:
		thread.start()
	for thread in threads:
		thread.join()

	return hwe