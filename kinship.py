"""
Kinship estimator using genotype likelihoods based on PC-Relate.
"""

__author__ = "Jonas Meisner"

# Import libraries
import numpy as np
import threading
from numba import jit
from math import sqrt


##### Functions #####
# Estimate diagonal entries of kinship matrix
@jit("void(f4[:, :], f4[:, :], i8, i8, f8[:])", nopython=True, nogil=True, cache=True)
def diagKinship(likeMatrix, Pi, S, N, diagK):
	m, n = Pi.shape # Dimensions

	for ind in xrange(S, min(S+N, m)):
		num = 0.0
		dem = 0.0

		# Estimate posterior probabilities and diagonal of kinship matrix
		for s in xrange(n):
			p0 = likeMatrix[3*ind, s]*(1 - Pi[ind, s])*(1 - Pi[ind, s])
			p1 = likeMatrix[3*ind+1, s]*2*Pi[ind, s]*(1 - Pi[ind, s])
			p2 = likeMatrix[3*ind+2, s]*Pi[ind, s]*Pi[ind, s]
			pSum = p0 + p1 + p2

			num += (0 - 2*Pi[ind, s])*(0 - 2*Pi[ind, s])*(p0/pSum)
			num += (1 - 2*Pi[ind, s])*(1 - 2*Pi[ind, s])*(p1/pSum)
			num += (2 - 2*Pi[ind, s])*(2 - 2*Pi[ind, s])*(p2/pSum)
			dem += Pi[ind, s]*(1-Pi[ind, s])
		diagK[ind] = num/(4*dem)

# Prepare numerator matrix
@jit("void(f4[:, :], f4[:, :], i8, i8, f8[:, :])", nopython=True, nogil=True, cache=True)
def numeratorKin(expG, Pi, S, N, X):
	m, n = expG.shape
	for ind in xrange(S, min(S+N, m)):
		for s in xrange(n):
			X[ind, s] = expG[ind, s] - 2*Pi[ind, s]

# Prepare denominator matrix
@jit("void(f4[:, :], i8, i8, f8[:, :])", nopython=True, nogil=True, cache=True)
def denominatorKin(Pi, S, N, X):
	m, n = Pi.shape
	for ind in xrange(S, min(S+N, m)):
		for s in xrange(n):
			X[ind, s] = sqrt(Pi[ind, s]*(1 - Pi[ind, s]))

# Estimate full kinship matrix with biased diagonal
def estimateKinship(expG, Pi, chunks, chunk_N):
	m, n = expG.shape
	X = np.empty((m, n))

	# Multithreading
	threads = [threading.Thread(target=numeratorKin, args=(expG, Pi, chunk, chunk_N, X)) for chunk in chunks]
	for thread in threads:
		thread.start()
	for thread in threads:
		thread.join()

	num = np.dot(X, X.T)

	# Multithreading
	threads = [threading.Thread(target=denominatorKin, args=(Pi, chunk, chunk_N, X)) for chunk in chunks]
	for thread in threads:
		thread.start()
	for thread in threads:
		thread.join()

	dem = 4*np.dot(X, X.T)
	return num/dem


def kinshipConomos(likeMatrix, Pi, expG, t=1):
	m, n = expG.shape
	diagK = np.empty(m)
	phi = np.zeros((m, m))

	# Multithreading parameters
	chunk_N = int(np.ceil(float(m)/t))
	chunks = [i * chunk_N for i in xrange(t)]

	# Multithreading
	threads = [threading.Thread(target=diagKinship, args=(likeMatrix, Pi, chunk, chunk_N, diagK)) for chunk in chunks]
	for thread in threads:
		thread.start()
	for thread in threads:
		thread.join()

	phi = estimateKinship(expG, Pi, chunks, chunk_N)
	np.fill_diagonal(phi, diagK) # Insert correct diagonal
	return phi