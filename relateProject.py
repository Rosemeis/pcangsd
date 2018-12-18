"""
Estimate individual allele frequencies for related individuals by projection.
"""

__author__ = "Jonas Meisner"

# Import libraries
import numpy as np
import threading
from numba import jit
from scipy.sparse.linalg import svds, eigsh
from math import sqrt
from covariance import updateFumagalli, updatePCAngsd, expGcenter, addIntercept, covPCAngsd, normalizeGeno, estimateCov
from emMAF import *
from helpFunctions import rmse2d_float32

##### Functions #####
# Estimate individual allele frequencies for unrelated individuals
def estimateF_unrelated(E, f, e, Pi, chunks, chunk_N):
	# Multithreading - Centering genotype dosages
	threads = [threading.Thread(target=expGcenter, args=(E, f, chunk, chunk_N)) for chunk in chunks]
	for thread in threads:
		thread.start()
	for thread in threads:
		thread.join()

	# Reduced SVD of rank K (Scipy library)
	W, s, U = svds(E, k=e)
	Pi = np.dot(W*s, U, out=Pi)

	# Multithreading - Adding intercept and clipping
	threads = [threading.Thread(target=addIntercept, args=(Pi, f, chunk, chunk_N)) for chunk in chunks]
	for thread in threads:
		thread.start()
	for thread in threads:
		thread.join()

	return W, U.T*s

# Estimate individual allele frequencies for related individuals
def estimateF_related(E, f, US, Pi, chunks, chunk_N):
	# Multithreading - Centering genotype dosages
	threads = [threading.Thread(target=expGcenter, args=(E, f, chunk, chunk_N)) for chunk in chunks]
	for thread in threads:
		thread.start()
	for thread in threads:
		thread.join()

	# Multiple linear regression model
	W = np.dot(np.linalg.inv(np.dot(US.T, US)), np.dot(US.T, E.T)).T
	Pi = np.dot(W, US.T, out=Pi)

	# Multithreading - Adding intercept and clipping
	threads = [threading.Thread(target=addIntercept, args=(Pi, f, chunk, chunk_N)) for chunk in chunks]
	for thread in threads:
		thread.start()
	for thread in threads:
		thread.join()

	return W

# Modified version of PCAngsd for accounting for relatedness
def relatedPCAngsd(likeMatrix, e, f, K, r_tole, M, M_tole, t=1):
	m, n = likeMatrix.shape # Dimension of likelihood matrix
	m /= 3 # Number of individuals

	# Boolean vector of unrelated individuals
	print "Masking related individuals with pair-wise kinship estimates >= " + str(r_tole) + "."
	K[np.triu_indices(m)] = 0 # Setting half of matrix to 0
	if len(np.unique(np.where(K > r_tole)[0])) < len(np.unique(np.where(K > r_tole)[1])):
		relatedIndices = np.unique(np.where(K > r_tole)[0])
	else:
		relatedIndices = np.unique(np.where(K > r_tole)[1])
	unrelatedI = np.isin(np.arange(m), relatedIndices, invert=True)
	unrelatedM = sum(unrelatedI)
	relatedI = np.invert(unrelatedI)
	relatedM = sum(relatedI)
	assert unrelatedM != m, "All individuals are unrelated at specified threshold!"

	like_UR = likeMatrix[np.repeat(unrelatedI, 3)]
	like_R = likeMatrix[np.repeat(relatedI, 3)]

	# Clean-up
	del K, relatedIndices

	# Reestimate population allele frequencies using unrelated individuals
	print "Re-estimating population allele frequencies using unrelated individuals."
	fUR = alleleEM(like_UR, t=t)

	# Multithreading parameters
	chunk_N_UR = int(np.ceil(float(unrelatedM)/t))
	chunks_UR = [i * chunk_N_UR for i in xrange(t)]
	chunk_N_R = int(np.ceil(float(relatedM)/t))
	chunks_R = [i * chunk_N_R for i in xrange(t)]

	# PCAngsd unrelated
	print "\nEstimating individual allele frequencies for " + str(unrelatedM) + " unrelated individuals."

	# Estimate individual allele frequencies for unrelated individuals
	E_UR = np.empty((unrelatedM, n), dtype=np.float32)

	# Multithreading
	threads = [threading.Thread(target=updateFumagalli, args=(like_UR, fUR, chunk, chunk_N_UR, E_UR)) for chunk in chunks_UR]
	for thread in threads:
		thread.start()
	for thread in threads:
		thread.join()

	Pi_UR = np.empty((unrelatedM, n), dtype=np.float32)
	W, US = estimateF_unrelated(E_UR, fUR, e, Pi_UR, chunks_UR, chunk_N_UR)
	prevW = np.copy(W)
	print "Individual allele frequencies estimated (1)"

	# Iterative frequency estimation
	for iteration in xrange(2, M+1):
		# Multithreading
		threads = [threading.Thread(target=updatePCAngsd, args=(like_UR, Pi_UR, chunk, chunk_N_UR, E_UR)) for chunk in chunks_UR]
		for thread in threads:
			thread.start()
		for thread in threads:
			thread.join()

		# Estimate individual allele frequencies
		W, US = estimateF_unrelated(E_UR, fUR, e, Pi_UR, chunks_UR, chunk_N_UR)

		# Break iterative update if converged
		diff = rmse2d_float32(np.absolute(W), np.absolute(prevW))
		print "Individual allele frequencies estimated (" + str(iteration) + "). RMSD=" + str(diff)
		if diff < M_tole:
			print "Estimation of individual allele frequencies for unrelated individuals has converged."
			break
		if iteration == 2:
			oldDiff = diff
		else:
			res = abs(diff - oldDiff)
			if res < 5e-7:
				print "Estimation of individual allele frequencies for unrelated individuals has converged due to small change in differences: " + str(res)
				break
			oldDiff = diff
		prevW = np.copy(W)

	# Estimate individual allele frequencies for related individuals
	print "\nEstimating individual allele frequencies for " + str(relatedM) + " related individuals."
	E_R = np.empty((relatedM, n), dtype=np.float32)

	# Multithreading
	threads = [threading.Thread(target=updateFumagalli, args=(like_R, fUR, chunk, chunk_N_R, E_R)) for chunk in chunks_R]
	for thread in threads:
		thread.start()
	for thread in threads:
		thread.join()

	Pi_R = np.empty((relatedM, n), dtype=np.float32)
	W = estimateF_related(E_R, fUR, US, Pi_R, chunks_R, chunk_N_R)
	prevW = np.copy(W)
	print "Individual allele frequencies estimated (1)"

	# Iterative frequency estimation
	for iteration in xrange(2, M+1):
		# Multithreading
		threads = [threading.Thread(target=updatePCAngsd, args=(like_R, Pi_R, chunk, chunk_N_R, E_R)) for chunk in chunks_R]
		for thread in threads:
			thread.start()
		for thread in threads:
			thread.join()

		# Estimate individual allele frequencies
		W = estimateF_related(E_R, fUR, US, Pi_R, chunks_R, chunk_N_R)

		# Break iterative update if converged
		diff = rmse2d_float32(np.absolute(W), np.absolute(prevW))
		print "Individual allele frequencies estimated (" + str(iteration) + "). RMSD=" + str(diff)
		if diff < M_tole:
			print "Estimation of individual allele frequencies for related individuals has converged."
			break
		if iteration == 2:
			oldDiff = diff
		else:
			res = abs(diff - oldDiff)
			if res < 5e-7:
				print "Estimation of individual allele frequencies for related individuals has converged due to small change in differences: " + str(res)
				break
			oldDiff = diff
		prevW = np.copy(W)

	del W, prevW

	# Multithreading
	threads = [threading.Thread(target=updatePCAngsd, args=(like_R, Pi_R, chunk, chunk_N_R, E_R)) for chunk in chunks_R]
	for thread in threads:
		thread.start()
	for thread in threads:
		thread.join()

	# Covariance matrix
	print "\nEstimating covariance matrix using unrelated individuals."
	diagC = np.empty(unrelatedM)

	# Multithreading
	threads = [threading.Thread(target=covPCAngsd, args=(like_UR, Pi_UR, fUR, chunk, chunk_N_UR, E_UR, diagC)) for chunk in chunks_UR]
	for thread in threads:
		thread.start()
	for thread in threads:
		thread.join()

	# Estimate covariance matrix (PCAngsd)
	C = estimateCov(E_UR, diagC, fUR, chunks_UR, chunk_N_UR)
	Sigma, V_UR = eigsh(C, k=e) # Eigendecomposition (Symmetric) - ARPACK
	Sigma = Sigma[::-1] # Sorted eigenvalues
	V_UR = V_UR[:, ::-1] # Sorted eigenvectors
	X = np.empty((unrelatedM, n))

	# Multithreading
	threads = [threading.Thread(target=normalizeGeno, args=(E_UR, fUR, chunk, chunk_N_UR, X)) for chunk in chunks_UR]
	for thread in threads:
		thread.start()
	for thread in threads:
		thread.join()

	# Projection step
	print "Projecting related individuals into PC-space."
	Z = np.dot(X.T, V_UR) # Projection onto eigenvectors - SNP weights
	X = np.empty((relatedM, n))

	# Multithreading
	threads = [threading.Thread(target=normalizeGeno, args=(E_R, fUR, chunk, chunk_N_R, X)) for chunk in chunks_R]
	for thread in threads:
		thread.start()
	for thread in threads:
		thread.join()

	V_R = np.dot(X, Z)/(Sigma*n)
	del X, Z

	# Combine unrelated and related individuals
	V = np.empty((m, e))
	V[unrelatedI] = V_UR
	V[relatedI] = V_R
	del V_UR, V_R
	E = np.empty((m, n), dtype=np.float32)
	E[unrelatedI] = E_UR
	E[relatedI] = E_R
	del E_UR, E_R
	Pi = np.empty((m, n), dtype=np.float32)
	Pi[unrelatedI] = Pi_UR
	Pi[relatedI] = Pi_R
	del Pi_UR, Pi_R
	return C, Pi, E, V, fUR