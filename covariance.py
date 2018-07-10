"""
Estimates the covariance matrix for NGS data using the PCAngsd method,
by estimating individual allele frequencies based on SVD.
"""

__author__ = "Jonas Meisner"

# Import libraries
import numpy as np
import threading
from numba import jit
from scipy.sparse.linalg import svds, eigsh
from math import sqrt
from helpFunctions import rmse2d_float32

##### Functions #####
# Update posterior expectations of the genotypes (Fumagalli method)
@jit("void(f4[:, :], f8[:], i8, i8, f4[:, :])", nopython=True, nogil=True, cache=True)
def updateFumagalli(likeMatrix, f, S, N, expG):
	m, n = likeMatrix.shape # Dimension of likelihood matrix
	m /= 3 # Number of individuals
	probMatrix = np.empty((3, n), dtype=np.float32)

	# Loop over individuals
	for ind in xrange(S, min(S+N, m)):
		# Estimate posterior probabilities
		for s in xrange(n):
			probMatrix[0, s] = likeMatrix[3*ind, s]*(1 - f[s])*(1 - f[s])
			probMatrix[1, s] = likeMatrix[3*ind+1, s]*2*f[s]*(1 - f[s])
			probMatrix[2, s] = likeMatrix[3*ind+2, s]*f[s]*f[s]
		probMatrix /= np.sum(probMatrix, axis=0)

		# Estimate genotype dosages
		for s in xrange(n):
			expG[ind, s] = 0.0
			for g in xrange(3):
				expG[ind, s] += probMatrix[g, s]*g

# Estimate posterior expecations of the genotypes and covariance matrix diagonal (Fumagalli method)
@jit("void(f4[:, :], f8[:], i8, i8, f4[:, :], f8[:])", nopython=True, nogil=True, cache=True)
def covFumagalli(likeMatrix, f, S, N, expG, diagC):
	m, n = likeMatrix.shape # Dimension of likelihood matrix
	m /= 3 # Number of individuals
	probMatrix = np.empty((3, n), dtype=np.float32)

	# Loop over individuals
	for ind in xrange(S, min(S+N, m)):
		# Estimate posterior probabilities
		for s in xrange(n):
			probMatrix[0, s] = likeMatrix[3*ind, s]*(1 - f[s])*(1 - f[s])
			probMatrix[1, s] = likeMatrix[3*ind+1, s]*2*f[s]*(1 - f[s])
			probMatrix[2, s] = likeMatrix[3*ind+2, s]*f[s]*f[s]
		probMatrix /= np.sum(probMatrix, axis=0)

		# Estimate genotype dosages and diagonal of GRM
		diagC[ind] = 0.0
		for s in xrange(n):
			expG[ind, s] = 0.0
			temp = 0.0
			for g in xrange(3):
				expG[ind, s] += probMatrix[g, s]*g
				temp += (g - 2*f[s])*(g - 2*f[s])*probMatrix[g, s]
			diagC[ind] += temp/(2*f[s]*(1 - f[s]))
		diagC[ind] /= n

# Update posterior expectations of the genotypes (PCAngsd)
@jit("void(f4[:, :], f4[:, :], i8, i8, f4[:, :])", nopython=True, nogil=True, cache=True)
def updatePCAngsd(likeMatrix, indF, S, N, expG):
	m, n = likeMatrix.shape # Dimension of likelihood matrix
	m /= 3 # Number of individuals
	probMatrix = np.empty((3, n), dtype=np.float32)

	# Loop over individuals
	for ind in xrange(S, min(S+N, m)):
		# Estimate posterior probabilities
		for s in xrange(n):
			probMatrix[0, s] = likeMatrix[3*ind, s]*(1 - indF[ind, s])*(1 - indF[ind, s])
			probMatrix[1, s] = likeMatrix[3*ind+1, s]*2*indF[ind, s]*(1 - indF[ind, s])
			probMatrix[2, s] = likeMatrix[3*ind+2, s]*indF[ind, s]*indF[ind, s]
		probMatrix /= np.sum(probMatrix, axis=0)

		# Estimate genotype dosages
		for s in xrange(n):
			expG[ind, s] = 0.0
			for g in xrange(3):
				expG[ind, s] += probMatrix[g, s]*g

# Estimate posterior expecations of the genotypes and covariance matrix diagonal (PCAngsd)
@jit("void(f4[:, :], f4[:, :], f8[:], i8, i8, f4[:, :], f8[:])", nopython=True, nogil=True, cache=True)
def covPCAngsd(likeMatrix, indF, f, S, N, expG, diagC):
	m, n = likeMatrix.shape # Dimension of likelihood matrix
	m /= 3 # Number of individuals
	probMatrix = np.empty((3, n), dtype=np.float32)

	for ind in xrange(S, min(S+N, m)):
		# Estimate posterior probabilities
		for s in xrange(n):
			probMatrix[0, s] = likeMatrix[3*ind, s]*(1 - indF[ind, s])*(1 - indF[ind, s])
			probMatrix[1, s] = likeMatrix[3*ind+1, s]*2*indF[ind, s]*(1 - indF[ind, s])
			probMatrix[2, s] = likeMatrix[3*ind+2, s]*indF[ind, s]*indF[ind, s]
		probMatrix /= np.sum(probMatrix, axis=0)

		# Estimate genotype dosages and diagonal of GRM
		diagC[ind] = 0.0
		for s in xrange(n):
			expG[ind, s] = 0.0
			temp = 0.0
			for g in xrange(3):
				expG[ind, s] += probMatrix[g, s]*g
				temp += (g - 2*f[s])*(g - 2*f[s])*probMatrix[g, s]
			diagC[ind] += temp/(2*f[s]*(1 - f[s]))
		diagC[ind] /= n

# Normalize the posterior expectations of the genotypes
@jit("void(f4[:, :], f8[:], i8, i8, f8[:, :])", nopython=True, nogil=True, cache=True)
def normalizeGeno(expG, f, S, N, X):
	m, n = expG.shape
	for ind in xrange(S, min(S+N, m)):
		for s in xrange(n):
			X[ind, s] = (expG[ind, s] - 2*f[s])/sqrt(2*f[s]*(1 - f[s]))

# Estimate covariance matrix
def estimateCov(expG, diagC, f, chunks, chunk_N):
	m, n = expG.shape
	X = np.empty((m, n))

	# Multithreading
	threads = [threading.Thread(target=normalizeGeno, args=(expG, f, chunk, chunk_N, X)) for chunk in chunks]
	for thread in threads:
		thread.start()
	for thread in threads:
		thread.join()

	C = np.dot(X, X.T)/n
	np.fill_diagonal(C, diagC)
	return C

# Center posterior expectations of the genotype for SVD
@jit("void(f4[:, :], f8[:], i8, i8)", nopython=True, nogil=True, cache=True)
def expGcenter(expG, f, S, N):
	m, n = expG.shape
	for ind in xrange(S, min(S+N, m)):
		for s in xrange(n):
			expG[ind, s] = expG[ind, s] - 2*f[s]

# Add intercept to reconstructed allele frequencies
@jit("void(f4[:, :], f8[:], i8, i8)", nopython=True, nogil=True, cache=True)
def addIntercept(indF, f, S, N):
	m, n = indF.shape
	for ind in xrange(S, min(S+N, m)):
		for s in xrange(n):
			indF[ind, s] += 2*f[s]
			indF[ind, s] /= 2
			indF[ind, s] = max(indF[ind, s], 1e-4)
			indF[ind, s] = min(indF[ind, s], 1-(1e-4))

# Estimate individual allele frequencies
def estimateF(expG, f, e, chunks, chunk_N):
	# Multithreading - Centering genotype dosages
	threads = [threading.Thread(target=expGcenter, args=(expG, f, chunk, chunk_N)) for chunk in chunks]
	for thread in threads:
		thread.start()
	for thread in threads:
		thread.join()

	# Reduced SVD of rank K (Scipy library)
	W, s, U = svds(expG, k=e)
	F = np.dot(W*s, U)

	# Multithreading - Adding intercept and clipping
	threads = [threading.Thread(target=addIntercept, args=(F, f, chunk, chunk_N)) for chunk in chunks]
	for thread in threads:
		thread.start()
	for thread in threads:
		thread.join()

	return F, W


##### PCAngsd #####
def PCAngsd(likeMatrix, EVs, M, f, M_tole, threads=1):
	m, n = likeMatrix.shape # Dimension of likelihood matrix
	m /= 3 # Number of individuals
	e = EVs

	# Multithreading parameters
	chunk_N = int(np.ceil(float(m)/threads))
	chunks = [i * chunk_N for i in xrange(threads)]

	# Initiate matrices
	expG = np.empty((m, n), dtype=np.float32)
	diagC = np.empty(m)

	# Estimate covariance matrix (Fumagalli) and infer number of PCs
	if EVs == 0:
		# Multithreading
		threads = [threading.Thread(target=covFumagalli, args=(likeMatrix, f, chunk, chunk_N, expG, diagC)) for chunk in chunks]
		for thread in threads:
			thread.start()
		for thread in threads:
			thread.join()

		# Estimate covariance matrix (Fumagalli)
		C = estimateCov(expG, diagC, f, chunks, chunk_N)
		if M == 0:
			print "Returning with ngsTools covariance matrix!"
			return C, None, e, expG

		# Velicer's Minimum Average Partial (MAP) Test
		eigVals, eigVecs = eigsh(C, k=15) # Eigendecomposition (Symmetric - Scipy library)
		sort = np.argsort(eigVals)[::-1] # Sorting vector
		eigVals = eigVals[sort] # Sorted eigenvalues
		eigVals[eigVals < 0] = 0
		eigVecs = eigVecs[:, sort] # Sorted eigenvectors
		loadings = eigVecs*np.sqrt(eigVals)
		mapTest = np.empty(15)

		# Loop over m-1 eigenvalues for MAP test (Shriner implementation)
		for eig in xrange(15):
			partcov = C - (np.dot(loadings[:, 0:(eig + 1)], loadings[:, 0:(eig + 1)].T))
			d = np.diag(partcov)

			if (np.sum(np.isnan(d)) > 0) or (np.sum(d == 0) > 0) or (np.sum(d < 0) > 0):
				mapTest[eig] = 1
			else:
				d = np.diagflat(1/np.sqrt(d))
				pr = np.dot(d, np.dot(partcov, d))
				mapTest[eig] = (np.sum(pr**2) - m)/(m*(m - 1))

		e = max([1, np.argmin(mapTest) + 1]) # Number of principal components retained
		print "Using " + str(e) + " principal components (MAP test)"

		# Release memory
		del eigVals, eigVecs, loadings, partcov, mapTest

	else:
		print "Using " + str(e) + " principal components (manually selected)"

		# Multithreading
		threads = [threading.Thread(target=updateFumagalli, args=(likeMatrix, f, chunk, chunk_N, expG)) for chunk in chunks]
		for thread in threads:
			thread.start()
		for thread in threads:
			thread.join()

	# Estimate individual allele frequencies
	predF, W = estimateF(expG, f, e, chunks, chunk_N)
	prevW = np.copy(W)
	print "Individual allele frequencies estimated (1)"

	# Iterative covariance estimation
	for iteration in xrange(2, M+1):
		# Multithreading
		threads = [threading.Thread(target=updatePCAngsd, args=(likeMatrix, predF, chunk, chunk_N, expG)) for chunk in chunks]
		for thread in threads:
			thread.start()
		for thread in threads:
			thread.join()

		# Estimate individual allele frequencies
		predF, W = estimateF(expG, f, e, chunks, chunk_N)

		# Break iterative update if converged
		diff = rmse2d_float32(np.absolute(W), np.absolute(prevW))
		print "Individual allele frequencies estimated (" + str(iteration) + "). RMSD=" + str(diff)
		if diff < M_tole:
			print "Estimation of individual allele frequencies has converged."
			break
		if iteration == 2:
			oldDiff = diff
		else:
			res = abs(diff - oldDiff)
			if res < 5e-7:
				print "Estimation of individual allele frequencies has converged due to small change in differences: " + str(res)
				break
			oldDiff = diff
		prevW = np.copy(W)

	# Multithreading
	threads = [threading.Thread(target=covPCAngsd, args=(likeMatrix, predF, f, chunk, chunk_N, expG, diagC)) for chunk in chunks]
	for thread in threads:
		thread.start()
	for thread in threads:
		thread.join()

	# Estimate covariance matrix (PCAngsd)
	C = estimateCov(expG, diagC, f, chunks, chunk_N)
	return C, predF, e, expG