"""
PCAngsd.
Estimates the covariance matrix for NGS data and individual allele frequencies.
"""

__author__ = "Jonas Meisner"

# Libraries
import numpy as np
from pcangsd import covariance_cy
from pcangsd import shared_cy
from scipy.sparse.linalg import eigsh, svds

##### PCAngsd #####
# Flip signs of SVD output - Based on scikit-learn
def signFlip(U, Vt):
	maxCols = np.argmax(np.abs(U), axis=0)
	signs = np.sign(U[maxCols, range(U.shape[1])])
	U *= signs
	Vt *= signs[:, np.newaxis]
	return U, Vt

# Estimate individual allele frequencies
def estimatePi(E, K, P):
	U, S, Vt = svds(E, k=K)
	U, Vt = signFlip(U, Vt)
	np.dot(U, S.reshape(-1,1)*Vt, out=P)
	return P, Vt

### PCAngsd iterations ###
def emPCA(L, f, e, iter, tole, t):
	m = L.shape[0]
	n = L.shape[1]//2

	# Initiate matrices
	E = np.zeros((m, n), dtype=np.float32)
	P = np.zeros((m, n), dtype=np.float32)
	dCov = np.zeros(n, dtype=np.float32)

	# Estimate covariance matrix (Fumagalli) and infer number of PCs
	if (e == 0) or (iter == 0):
		# Prepare dosages and diagonal
		covariance_cy.covNormal(L, f, E, dCov, t)
		C = np.dot(E.T, E)/float(m)
		np.fill_diagonal(C, dCov/float(m))

		if iter == 0:
			print("Returning with ngsTools covariance matrix!")
			return C, None, None, 0, False

		# Velicer's Minimum Average Partial (MAP) Test
		eVal, eVec = eigsh(C, k=min(n-1, 15)) # Eigendecomposition
		eVal = eVal[::-1] # Sorted eigenvalues
		eVal[eVal < 0] = 0
		eVec = eVec[:,::-1] # Sorted eigenvectors
		loading = eVec*np.sqrt(eVal)
		mapTest = np.empty(min(n-1, 15), dtype=np.float32)

		# Loop over m-1 eigenvalues for MAP test (Shriner implementation)
		for eig in range(min(n-1, 15)):
			partcov = C - (np.dot(loading[:,0:(eig + 1)], loading[:,0:(eig + 1)].T))
			d = np.diag(partcov)

			if (np.sum(np.isnan(d)) > 0) or (np.sum(d == 0) > 0) or (np.sum(d < 0) > 0):
				mapTest[eig] = 1.0
			else:
				d = np.diagflat(1.0/np.sqrt(d))
				pr = np.dot(d, np.dot(partcov, d))
				mapTest[eig] = (np.sum(pr**2) - n)/(n*(n - 1))

		K = max([1, np.argmin(mapTest) + 1]) # Number of principal components retained
		print(f"Using {K} principal components (MAP test).")
		del d, eVal, eVec, loading, mapTest, partcov
		if 'pr' in vars():
			del pr
	else:
		K = e
		print(f"Using {K} principal components (manually selected).")

	# Estimate individual allele frequencies
	covariance_cy.updateNormal(L, f, E, t)
	P, Vt = estimatePi(E, K, P)
	print("Individual allele frequencies estimated (1).")

	# Iterative estimation
	for it in range(iter):
		prevV = np.copy(Vt)
		covariance_cy.updatePCAngsd(L, f, P, E, t)
		P, Vt = estimatePi(E, K, P)
		# Check for convergence
		diff = shared_cy.rmse2d(Vt, prevV)
		print("Individual allele frequencies estimated " + \
			f"({it+2}).\tRMSE={np.round(diff,9)}")
		if diff < tole:
			print("PCAngsd converged.")
			converged = True
			break
		if it == (iter - 1):
			print("PCAngsd did not converge!")
			converged = False
	del Vt, prevV

	# Estimate final covariance matrix
	dCov.fill(0.0)
	covariance_cy.covPCAngsd(L, f, P, E, dCov, t)
	C = np.dot(E.T, E)/float(m)
	np.fill_diagonal(C, dCov/float(m))
	del E, dCov # Release memory
	return C, P, K, it, converged
