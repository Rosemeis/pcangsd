"""
Help functions to use in the PCAngsd framework.
"""

__author__ = "Jonas Meisner"

# Import libraries
import numpy as np
import threading
import os
from numba import jit
from math import sqrt

# Read Beagle gzip file
def readGzipBeagle(beagle, n):
	with os.popen("zcat " + str(beagle)) as f:
		c = -1
		for line in f:
			if c < 0:
				c += 1
				m3 = len(line.split("\t"))-3
				if n == 0:
					n = 1000000
				likeMatrix = np.empty((n, m3), dtype=np.float32)
			else:
				if c == n:
					n *= 2
					likeMatrix.resize((n, m3), refcheck=False)
					print "Changed allocation to " + str(n) + " sites"
				likeMatrix[c, :] = line.split("\t")[3:]
				c += 1
		return likeMatrix[:c, :].T


# Root mean squared error
@jit("f8(f8[:], f8[:])", nopython=True, nogil=True, cache=True)
def rmse1d(A, B):
	sumA = 0.0
	for i in xrange(A.shape[0]):
		sumA += (A[i] - B[i])*(A[i] - B[i])
	sumA /= (A.shape[0])
	return sqrt(sumA)

# Multi-threaded RMSE
@jit("void(f4[:, :], f4[:, :], i8, i8, f8[:])", nopython=True, nogil=True, cache=True)
def rmse2d_inner_float32(A, B, S, N, V):
	m, n = A.shape
	for i in xrange(S, min(S+N, m)):
		for j in xrange(n):
			V[i] += (A[i, j] - B[i, j])*(A[i, j] - B[i, j])

def rmse2d_multi_float32(A, B, chunks, chunk_N):
	m, n = A.shape
	sumA = np.zeros(m)

	# Multithreading
	threads = [threading.Thread(target=rmse2d_inner_float32, args=(A, B, chunk, chunk_N, sumA)) for chunk in chunks]
	for thread in threads:
		thread.start()
	for thread in threads:
		thread.join()

	return sqrt(np.sum(sumA)/(m*n))

@jit("void(f8[:, :], f8[:, :], i8, i8, f8[:])", nopython=True, nogil=True, cache=True)
def rmse2d_inner(A, B, S, N, V):
	m, n = A.shape
	for i in xrange(S, min(S+N, m)):
		for j in xrange(n):
			V[i] += (A[i, j] - B[i, j])*(A[i, j] - B[i, j])

def rmse2d_multi(A, B, chunks, chunk_N):
	m, n = A.shape
	sumA = np.zeros(m)

	# Multithreading
	threads = [threading.Thread(target=rmse2d_inner, args=(A, B, chunk, chunk_N, sumA)) for chunk in chunks]
	for thread in threads:
		thread.start()
	for thread in threads:
		thread.join()

	return sqrt(np.sum(sumA)/(m*n))

# Root mean squared error
@jit("f8(f8[:, :], f8[:, :])", nopython=True, nogil=True, cache=True)
def rmse2d(A, B):
	sumA = 0.0
	for i in xrange(A.shape[0]):
		for j in xrange(A.shape[1]):
			sumA += (A[i, j] - B[i, j])*(A[i, j] - B[i, j])
	sumA /= (A.shape[0]*A.shape[1])
	return sqrt(sumA)

# Root mean squared error
@jit("f8(f4[:, :], f4[:, :])", nopython=True, nogil=True, cache=True)
def rmse2d_float32(A, B):
	sumA = 0.0
	for i in xrange(A.shape[0]):
		for j in xrange(A.shape[1]):
			sumA += (A[i, j] - B[i, j])*(A[i, j] - B[i, j])
	sumA /= (A.shape[0]*A.shape[1])
	return sqrt(sumA)

# Multi-threaded frobenius
@jit("void(f4[:, :], f4[:, :], i8, i8, f8[:])", nopython=True, nogil=True, cache=True)
def frobenius2d_inner(A, B, S, N, V):
	m, n = A.shape
	for i in xrange(S, min(S+N, m)):
		for j in xrange(n):
			V[i] += (A[i, j] - B[i, j])*(A[i, j] - B[i, j])

def frobenius2d_multi(A, B, chunks, chunk_N):
	m, n = A.shape
	sumA = np.zeros(m)

	# Multithreading
	threads = [threading.Thread(target=frobenius2d_inner, args=(A, B, chunk, chunk_N, sumA)) for chunk in chunks]
	for thread in threads:
		thread.start()
	for thread in threads:
		thread.join()

	return sqrt(np.sum(sumA))

# Frobenius norm
@jit("f8(f4[:, :], f4[:, :])", nopython=True, nogil=True, cache=True)
def frobenius(A, B):
	sumA = 0.0
	for i in xrange(A.shape[0]):
		for j in xrange(A.shape[1]):
			sumA += (A[i, j] - B[i, j])*(A[i, j] - B[i, j])
	return sqrt(sumA)

# Parser for PLINK files
def readPlink(plink, epsilon, t):
	from pysnptools.snpreader import Bed # Import Microsoft Genomics PLINK reader
	snpClass = Bed(plink, count_A1=True) # Create PLINK instance
	pos = np.copy(snpClass.sid) # Save variant IDs
	snpFile = snpClass.read(dtype=np.float32) # Read PLINK files into memory
	m, _ = snpFile.val.shape
	f = np.nanmean(snpFile.val, axis=0, dtype=np.float64)/2 # Allele frequencies

	# Construct genotype likelihood matrix
	print "Converting PLINK files into genotype likelihood matrix"
	likeMatrix = np.zeros((3*m, snpFile.val.shape[1]), dtype=np.float32)
	chunk_N = int(np.ceil(float(m)/t))
	chunks = [i * chunk_N for i in xrange(t)]

	# Multithreading
	threads = [threading.Thread(target=convertPlink, args=(likeMatrix, snpFile.val, chunk, chunk_N, epsilon)) for chunk in chunks]
	for thread in threads:
		thread.start()
	for thread in threads:
		thread.join()

	return likeMatrix, f, pos

# Convert PLINK genotype matrix into genotype likelihoods
@jit("void(f4[:, :], f4[:, :], i8, i8, f8)", nopython=True, nogil=True, cache=True)
def convertPlink(likeMatrix, G, S, N, epsilon):
	m, n = G.shape # Dimension of genotype matrix
	for ind in xrange(S, min(S+N, m)):
		for s in xrange(n):
			if np.isnan(G[ind, s]): # Missing site
				likeMatrix[3*ind, s] = 0.333333
				likeMatrix[3*ind+1, s] = 0.333333
				likeMatrix[3*ind+2, s] = 0.333333
			else:
				for g in xrange(3):
					if int(G[ind, s]) == g:
						likeMatrix[3*ind + g, s] = 1.0 - epsilon
					else:
						likeMatrix[3*ind + g, s] = epsilon/2.0
