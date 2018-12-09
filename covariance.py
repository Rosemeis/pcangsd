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
def updateFumagalli(likeMatrix, f, S, N, E):
	m, n = likeMatrix.shape # Dimension of likelihood matrix
	m /= 3 # Number of individuals

	for ind in xrange(S, min(S+N, m)):
		# Estimate posterior probabilities
		for s in xrange(n):
			p0 = likeMatrix[3*ind, s]*(1 - f[s])*(1 - f[s])
			p1 = likeMatrix[3*ind+1, s]*2*f[s]*(1 - f[s])
			p2 = likeMatrix[3*ind+2, s]*f[s]*f[s]
			pSum = p0 + p1 + p2

			# Update dosage
			E[ind, s] = (p1 + 2*p2)/pSum

# Estimate posterior expecations of the genotypes and covariance matrix diagonal (Fumagalli method)
@jit("void(f4[:, :], f8[:], i8, i8, f4[:, :], f8[:])", nopython=True, nogil=True, cache=True)
def covFumagalli(likeMatrix, f, S, N, E, diagC):
	m, n = likeMatrix.shape # Dimension of likelihood matrix
	m /= 3 # Number of individuals

	for ind in xrange(S, min(S+N, m)):
		diagC[ind] = 0.0

		# Estimate posterior probabilities
		for s in xrange(n):
			p0 = likeMatrix[3*ind, s]*(1 - f[s])*(1 - f[s])
			p1 = likeMatrix[3*ind+1, s]*2*f[s]*(1 - f[s])
			p2 = likeMatrix[3*ind+2, s]*f[s]*f[s]
			pSum = p0 + p1 + p2

			# Update dosage and diagonal of GRM
			E[ind, s] = (p1 + 2*p2)/pSum
			temp = (0 - 2*f[s])*(0 - 2*f[s])*p0
			temp += (1 - 2*f[s])*(1 - 2*f[s])*p1
			temp += (2 - 2*f[s])*(2 - 2*f[s])*p2
			diagC[ind] += temp/(2*f[s]*(1 - f[s]))
		diagC[ind] /= n

# Update posterior expectations of the genotypes (PCAngsd)
@jit("void(f4[:, :], f4[:, :], i8, i8, f4[:, :])", nopython=True, nogil=True, cache=True)
def updatePCAngsd(likeMatrix, Pi, S, N, E):
	m, n = Pi.shape # Dimensions

	for ind in xrange(S, min(S+N, m)):
		# Estimate posterior probabilities
		for s in xrange(n):
			p0 = likeMatrix[3*ind, s]*(1 - Pi[ind, s])*(1 - Pi[ind, s])
			p1 = likeMatrix[3*ind+1, s]*2*Pi[ind, s]*(1 - Pi[ind, s])
			p2 = likeMatrix[3*ind+2, s]*Pi[ind, s]*Pi[ind, s]
			pSum = p0 + p1 + p2

			# Update dosage
			E[ind, s] = (p1 + 2*p2)/pSum

# Estimate posterior expecations of the genotypes and covariance matrix diagonal (PCAngsd)
@jit("void(f4[:, :], f4[:, :], f8[:], i8, i8, f4[:, :], f8[:])", nopython=True, nogil=True, cache=True)
def covPCAngsd(likeMatrix, Pi, f, S, N, E, diagC):
	m, n = Pi.shape # Dimensions

	for ind in xrange(S, min(S+N, m)):
		diagC[ind] = 0.0

		# Estimate posterior probabilities
		for s in xrange(n):
			p0 = likeMatrix[3*ind, s]*(1 - Pi[ind, s])*(1 - Pi[ind, s])
			p1 = likeMatrix[3*ind+1, s]*2*Pi[ind, s]*(1 - Pi[ind, s])
			p2 = likeMatrix[3*ind+2, s]*Pi[ind, s]*Pi[ind, s]
			pSum = p0 + p1 + p2

			# Update dosage
			E[ind, s] = (p1 + 2*p2)/pSum
			temp = (0 - 2*f[s])*(0 - 2*f[s])*p0
			temp += (1 - 2*f[s])*(1 - 2*f[s])*p1
			temp += (2 - 2*f[s])*(2 - 2*f[s])*p2
			diagC[ind] += temp/(2*f[s]*(1 - f[s]))
		diagC[ind] /= n

# Normalize the posterior expectations of the genotypes
@jit("void(f4[:, :], f8[:], i8, i8, f8[:, :])", nopython=True, nogil=True, cache=True)
def normalizeGeno(E, f, S, N, X):
	m, n = E.shape
	for ind in xrange(S, min(S+N, m)):
		for s in xrange(n):
			X[ind, s] = (E[ind, s] - 2*f[s])/sqrt(2*f[s]*(1 - f[s]))

# Estimate covariance matrix
def estimateCov(E, diagC, f, chunks, chunk_N):
	m, n = E.shape
	X = np.empty((m, n))

	# Multithreading
	threads = [threading.Thread(target=normalizeGeno, args=(E, f, chunk, chunk_N, X)) for chunk in chunks]
	for thread in threads:
		thread.start()
	for thread in threads:
		thread.join()

	C = np.dot(X, X.T)/n
	np.fill_diagonal(C, diagC)
	return C

# Center posterior expectations of the genotype for SVD
@jit("void(f4[:, :], f8[:], i8, i8)", nopython=True, nogil=True, cache=True)
def expGcenter(E, f, S, N):
	m, n = E.shape
	for ind in xrange(S, min(S+N, m)):
		for s in xrange(n):
			E[ind, s] -= 2*f[s]

# Add intercept to reconstructed allele frequencies
@jit("void(f4[:, :], f8[:], i8, i8)", nopython=True, nogil=True, cache=True)
def addIntercept(Pi, f, S, N):
	m, n = Pi.shape
	for ind in xrange(S, min(S+N, m)):
		for s in xrange(n):
			Pi[ind, s] += 2*f[s]
			Pi[ind, s] /= 2
			Pi[ind, s] = max(Pi[ind, s], 1e-4)
			Pi[ind, s] = min(Pi[ind, s], 1-(1e-4))

# Estimate individual allele frequencies
def estimateF(E, f, e, F, chunks, chunk_N):
	# Multithreading - Centering genotype dosages
	threads = [threading.Thread(target=expGcenter, args=(E, f, chunk, chunk_N)) for chunk in chunks]
	for thread in threads:
		thread.start()
	for thread in threads:
		thread.join()

	# Reduced SVD of rank K (Scipy library)
	W, s, U = svds(E, k=e)
	F = np.dot(W*s, U, out=F)

	# Multithreading - Adding intercept and clipping
	threads = [threading.Thread(target=addIntercept, args=(F, f, chunk, chunk_N)) for chunk in chunks]
	for thread in threads:
		thread.start()
	for thread in threads:
		thread.join()

	return W


##### PCAngsd #####
def PCAngsd(likeMatrix, EVs, M, f, M_tole, threads=1):
	m, n = likeMatrix.shape # Dimension of likelihood matrix
	m /= 3 # Number of individuals
	e = EVs

	# Multithreading parameters
	chunk_N = int(np.ceil(float(m)/threads))
	chunks = [i * chunk_N for i in xrange(threads)]

	# Initiate matrices
	E = np.empty((m, n), dtype=np.float32)
	diagC = np.empty(m)

	# Estimate covariance matrix (Fumagalli) and infer number of PCs
	if EVs == 0:
		# Multithreading
		threads = [threading.Thread(target=covFumagalli, args=(likeMatrix, f, chunk, chunk_N, E, diagC)) for chunk in chunks]
		for thread in threads:
			thread.start()
		for thread in threads:
			thread.join()

		# Estimate covariance matrix (Fumagalli)
		C = estimateCov(E, diagC, f, chunks, chunk_N)
		if M == 0:
			print "Returning with ngsTools covariance matrix!"
			return C, None, e, E

		# Velicer's Minimum Average Partial (MAP) Test
		eigVals, eigVecs = eigsh(C, k=min(m-1, 15)) # Eigendecomposition (Symmetric - Scipy library)
		sort = np.argsort(eigVals)[::-1] # Sorting vector
		eigVals = eigVals[sort] # Sorted eigenvalues
		eigVals[eigVals < 0] = 0
		eigVecs = eigVecs[:, sort] # Sorted eigenvectors
		loadings = eigVecs*np.sqrt(eigVals)
		mapTest = np.empty(min(m-1, 15))

		# Loop over m-1 eigenvalues for MAP test (Shriner implementation)
		for eig in xrange(min(m-1, 15)):
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
		threads = [threading.Thread(target=updateFumagalli, args=(likeMatrix, f, chunk, chunk_N, E)) for chunk in chunks]
		for thread in threads:
			thread.start()
		for thread in threads:
			thread.join()

	# Estimate individual allele frequencies
	Pi = np.empty((m, n), dtype=np.float32)
	W = estimateF(E, f, e, Pi, chunks, chunk_N)
	prevW = np.copy(W)
	print "Individual allele frequencies estimated (1)"

	# Iterative covariance estimation
	for iteration in xrange(2, M+1):
		# Multithreading
		threads = [threading.Thread(target=updatePCAngsd, args=(likeMatrix, Pi, chunk, chunk_N, E)) for chunk in chunks]
		for thread in threads:
			thread.start()
		for thread in threads:
			thread.join()

		# Estimate individual allele frequencies
		W = estimateF(E, f, e, Pi, chunks, chunk_N)

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

	del W, prevW

	# Multithreading
	threads = [threading.Thread(target=covPCAngsd, args=(likeMatrix, Pi, f, chunk, chunk_N, E, diagC)) for chunk in chunks]
	for thread in threads:
		thread.start()
	for thread in threads:
		thread.join()

	# Estimate covariance matrix (PCAngsd)
	C = estimateCov(E, diagC, f, chunks, chunk_N)
	return C, Pi, e, E