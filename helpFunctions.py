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
def readGzipBeagle(beagle, nUser, post):
	with os.popen("zcat " + str(beagle)) as f:
		c = -1
		for line in f:
			if c < 0:
				c += 1
				m3 = len(line.split("\t"))-3
				if nUser == 0:
					n = 750000
				else:
					n = nUser
				likeMatrix = np.empty((n, m3), dtype=np.float32)

				if post:
					snpVector = np.chararray(n, itemsize=16)
					alleleMatrix = np.chararray((n, 2), itemsize=8)
			else:
				if c == n:
					n *= 2
					likeMatrix.resize((n, m3), refcheck=False)
					if post:
						snpVector.resize(n, refcheck=False)
						alleleMatrix.resize((n, 2), refcheck=False)
					print "Changed allocation to " + str(n) + " sites"
				temp = line.split("\t")
				likeMatrix[c, :] = temp[3:]
				if post:
					snpVector[c] = temp[0]
					alleleMatrix[c, :] = temp[1:3]
				c += 1

		if post:
			return likeMatrix[:c, :].T, snpVector[:c], alleleMatrix[:c]
		else:
			return likeMatrix[:c, :].T, None, None

# Write Beagle gzip file
def writeReadBeagle(postBeagle, postMatrix, snpVector=None, alleleMatrix=None, indList=None):
	m, n = postMatrix.shape
	with open(str(postBeagle), "wb") as f:
		if indList is None: # Given ANGSD beagle file
			headerList = ["marker", "allele1", "allele2"] + ["Ind" + str(i) for j in range(m/3) for i in [j, j, j]]
			f.write("\t".join(headerList) + "\n")
			for s in xrange(n):
				f.write(snpVector[s] + "\t")
				f.write(alleleMatrix[s, 0] + "\t")
				f.write(alleleMatrix[s, 1] + "\t")
				postMatrix[:, s].tofile(f, sep="\t", format="%.6f")
				f.write("\n")
		else:
			headerList = ["marker", "allele1", "allele2"] + [str(j) + "_" + str(i) for j in indList for i in [j, j, j]]
			f.write("\t".join(headerList) + "\n")
			for s in xrange(n):
				f.write(str(snpVector[s]) + "\t")
				f.write("\t".join(["0", "1"]))
				postMatrix[:, s].tofile(f, sep="\t", format="%.6f")
				f.write("\n")

# Root mean squared error
@jit("f8(f8[:], f8[:])", nopython=True, nogil=True, cache=True)
def rmse1d(A, B):
	sumA = 0.0
	for i in xrange(A.shape[0]):
		sumA += (A[i] - B[i])*(A[i] - B[i])
	sumA /= (A.shape[0])
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
	indList = np.copy(snpClass.iid[:, 1])
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

	return likeMatrix, f, pos, indList

# Convert PLINK genotype matrix into genotype likelihoods
@jit("void(f4[:, :], f4[:, :], i8, i8, f8)", nopython=True, nogil=True, cache=True)
def convertPlink(likeMatrix, G, S, N, e):
	m, n = G.shape # Dimension of genotype matrix
	for ind in xrange(S, min(S+N, m)):
		for s in xrange(n):
			if np.isnan(G[ind, s]): # Missing site
				likeMatrix[3*ind, s] = 0.333333
				likeMatrix[3*ind+1, s] = 0.333333
				likeMatrix[3*ind+2, s] = 0.333333
			else:
				if int(G[ind, s]) == 0:
					likeMatrix[3*ind, s] = (1 - e)*(1 - e)
					likeMatrix[3*ind + 1, s] = 2*(1 - e)*e
					likeMatrix[3*ind + 2, s] = e*e
				elif int(G[ind, s]) == 1:
					likeMatrix[3*ind, s] = (1 - e)*e
					likeMatrix[3*ind + 1, s] = (1 - e)*(1 - e) + e*e
					likeMatrix[3*ind + 2, s] = (1 - e)*e
				else:
					likeMatrix[3*ind, s] = e*e
					likeMatrix[3*ind + 1, s] = 2*(1 - e)*e
					likeMatrix[3*ind + 2, s] = (1 - e)*(1 - e)

# Alter likelihood matrix to posteriors
@jit("void(f4[:, :], f4[:, :], i8, i8)", nopython=True, nogil=True, cache=True)
def convertLikePost_inner(likeMatrix, Pi, S, N):
	m, n = likeMatrix.shape # Dimension of likelihood matrix
	m /= 3 # Number of individuals

	for ind in xrange(S, min(S+N, m)):
		# Estimate posterior probabilities
		for s in xrange(n):
			p0 = likeMatrix[3*ind, s]*(1 - Pi[ind, s])*(1 - Pi[ind, s])
			p1 = likeMatrix[3*ind+1, s]*2*Pi[ind, s]*(1 - Pi[ind, s])
			p2 = likeMatrix[3*ind+2, s]*Pi[ind, s]*Pi[ind, s]
			pSum = p0 + p1 + p2

			# Update matrix
			likeMatrix[3*ind, s] = p0/pSum
			likeMatrix[3*ind+1, s] = p1/pSum
			likeMatrix[3*ind+2, s] = p2/pSum

@jit("void(f4[:, :], f8[:], i8, i8)", nopython=True, nogil=True, cache=True)
def convertLikePost_innerNoIndF(likeMatrix, f, S, N):
	m, n = likeMatrix.shape # Dimension of likelihood matrix
	m /= 3 # Number of individuals

	for ind in xrange(S, min(S+N, m)):
		# Estimate posterior probabilities
		for s in xrange(n):
			p0 = likeMatrix[3*ind, s]*(1 - f[s])*(1 - f[s])
			p1 = likeMatrix[3*ind+1, s]*2*f[s]*(1 - f[s])
			p2 = likeMatrix[3*ind+2, s]*f[s]*f[s]
			pSum = p0 + p1 + p2

			# Update matrix
			likeMatrix[3*ind, s] = p0/pSum
			likeMatrix[3*ind+1, s] = p1/pSum
			likeMatrix[3*ind+2, s] = p2/pSum

def convertLikePost(likeMatrix, Pi, t):
	m, n = Pi.shape
	
	# Multithreading parameters
	chunk_N = int(np.ceil(float(m)/t))
	chunks = [i * chunk_N for i in xrange(t)]

	# Multithreading
	threads = [threading.Thread(target=convertLikePost_inner, args=(likeMatrix, Pi, chunk, chunk_N)) for chunk in chunks]
	for thread in threads:
		thread.start()
	for thread in threads:
		thread.join()

def convertLikePostNoIndF(likeMatrix, f, t):
	m, n = likeMatrix.shape
	m /= 3
	
	# Multithreading parameters
	chunk_N = int(np.ceil(float(m)/t))
	chunks = [i * chunk_N for i in xrange(t)]

	# Multithreading
	threads = [threading.Thread(target=convertLikePost_innerNoIndF, args=(likeMatrix, f, chunk, chunk_N)) for chunk in chunks]
	for thread in threads:
		thread.start()
	for thread in threads:
		thread.join()