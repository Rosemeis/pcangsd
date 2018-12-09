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
def estimateF_unrelated(expG, f, e, chunks, chunk_N):
	# Multithreading - Centering genotype dosages
	threads = [threading.Thread(target=expGcenter, args=(expG, f, chunk, chunk_N)) for chunk in chunks]
	for thread in threads:
		thread.start()
	for thread in threads:
		thread.join()

	# Reduced SVD of rank K (Scipy library)
	W, s, U = svds(expG, k=e)
	F = np.dot(W*s, U)
	US = U.T*s

	# Multithreading - Adding intercept and clipping
	threads = [threading.Thread(target=addIntercept, args=(F, f, chunk, chunk_N)) for chunk in chunks]
	for thread in threads:
		thread.start()
	for thread in threads:
		thread.join()

	return F, W, US

# Estimate individual allele frequencies for related individuals
def estimateF_related(expG, f, US, chunks, chunk_N):
	# Multithreading - Centering genotype dosages
	threads = [threading.Thread(target=expGcenter, args=(expG, f, chunk, chunk_N)) for chunk in chunks]
	for thread in threads:
		thread.start()
	for thread in threads:
		thread.join()

	# Multiple linear regression model
	WT = np.dot(np.linalg.inv(np.dot(US.T, US)), np.dot(US.T, expG.T))
	F = np.dot(US, WT).T

	# Multithreading - Adding intercept and clipping
	threads = [threading.Thread(target=addIntercept, args=(F, f, chunk, chunk_N)) for chunk in chunks]
	for thread in threads:
		thread.start()
	for thread in threads:
		thread.join()

	return F, WT.T

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
	assert unrelatedM != m, "All individuals are unrelated!"

	like_UR = likeMatrix[np.repeat(unrelatedI, 3)]
	like_R = likeMatrix[np.repeat(relatedI, 3)]

	# Clean-up
	del K, relatedIndices

	# Reestimate population allele frequencies using unrelated individuals
	print "Re-estimating population allele frequencies using unrelated individuals."
	fUR = alleleEM(like_UR, threads=t)

	# Multithreading parameters
	chunk_N_UR = int(np.ceil(float(unrelatedM)/t))
	chunks_UR = [i * chunk_N_UR for i in xrange(t)]
	chunk_N_R = int(np.ceil(float(relatedM)/t))
	chunks_R = [i * chunk_N_R for i in xrange(t)]

	# PCAngsd unrelated
	print "\nEstimating individual allele frequencies for " + str(unrelatedM) + " unrelated individuals."

	# Estimate individual allele frequencies for unrelated individuals
	expG_UR = np.empty((unrelatedM, n), dtype=np.float32)

	# Multithreading
	threads = [threading.Thread(target=updateFumagalli, args=(like_UR, fUR, chunk, chunk_N_UR, expG_UR)) for chunk in chunks_UR]
	for thread in threads:
		thread.start()
	for thread in threads:
		thread.join()

	predF_UR, W, US = estimateF_unrelated(expG_UR, fUR, e, chunks_UR, chunk_N_UR)
	prevW = np.copy(W)
	print "Individual allele frequencies estimated (1)"

	# Iterative frequency estimation
	for iteration in xrange(2, M+1):
		# Multithreading
		threads = [threading.Thread(target=updatePCAngsd, args=(like_UR, predF_UR, chunk, chunk_N_UR, expG_UR)) for chunk in chunks_UR]
		for thread in threads:
			thread.start()
		for thread in threads:
			thread.join()

		# Estimate individual allele frequencies
		predF_UR, W, US = estimateF_unrelated(expG_UR, fUR, e, chunks_UR, chunk_N_UR)

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
	expG_R = np.empty((relatedM, n), dtype=np.float32)

	# Multithreading
	threads = [threading.Thread(target=updateFumagalli, args=(like_R, fUR, chunk, chunk_N_R, expG_R)) for chunk in chunks_R]
	for thread in threads:
		thread.start()
	for thread in threads:
		thread.join()

	predF_R, W = estimateF_related(expG_R, fUR, US, chunks_R, chunk_N_R)
	prevW = np.copy(W)
	print "Individual allele frequencies estimated (1)"

	# Iterative frequency estimation
	for iteration in xrange(2, M+1):
		# Multithreading
		threads = [threading.Thread(target=updatePCAngsd, args=(like_R, predF_R, chunk, chunk_N_R, expG_R)) for chunk in chunks_R]
		for thread in threads:
			thread.start()
		for thread in threads:
			thread.join()

		# Estimate individual allele frequencies
		predF_R, W = estimateF_related(expG_R, fUR, US, chunks_R, chunk_N_R)

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
	threads = [threading.Thread(target=updatePCAngsd, args=(like_R, predF_R, chunk, chunk_N_R, expG_R)) for chunk in chunks_R]
	for thread in threads:
		thread.start()
	for thread in threads:
		thread.join()

	# Covariance matrix
	print "\nEstimating covariance matrix using unrelated individuals."
	diagC = np.empty(unrelatedM)

	# Multithreading
	threads = [threading.Thread(target=covPCAngsd, args=(like_UR, predF_UR, fUR, chunk, chunk_N_UR, expG_UR, diagC)) for chunk in chunks_UR]
	for thread in threads:
		thread.start()
	for thread in threads:
		thread.join()

	# Estimate covariance matrix (PCAngsd)
	C = estimateCov(expG_UR, diagC, fUR, chunks_UR, chunk_N_UR)
	Sigma, V_UR = eigsh(C, k=e) # Eigendecomposition (Symmetric - Scipy library)
	sort = np.argsort(Sigma)[::-1] # Sorting vector
	Sigma = Sigma[sort] # Sorted eigenvalues
	V_UR = V_UR[:, sort] # Sorted eigenvectors
	X = np.empty((unrelatedM, n))

	# Multithreading
	threads = [threading.Thread(target=normalizeGeno, args=(expG_UR, fUR, chunk, chunk_N_UR, X)) for chunk in chunks_UR]
	for thread in threads:
		thread.start()
	for thread in threads:
		thread.join()

	Z = np.dot(X.T, V_UR)

	# Projection
	print "Projecting related individuals into PC-space."
	X = np.empty((relatedM, n))

	# Multithreading
	threads = [threading.Thread(target=normalizeGeno, args=(expG_R, fUR, chunk, chunk_N_R, X)) for chunk in chunks_R]
	for thread in threads:
		thread.start()
	for thread in threads:
		thread.join()

	V_R = np.dot(X, Z)/(Sigma*n)
	del X

	# Combine dosages and frequencies
	expG = np.empty((m, n), dtype=np.float32)
	expG[unrelatedI] = expG_UR
	expG[relatedI] = expG_R
	del expG_UR, expG_R
	predF = np.empty((m, n), dtype=np.float32)
	predF[unrelatedI] = predF_UR
	predF[relatedI] = predF_R
	del predF_UR, predF_R
	return C, predF, expG, np.vstack((V_UR, V_R)), Sigma, fUR