import numpy as np
from pcangsd import covariance_cy
from pcangsd import shared_cy
from scipy.sparse.linalg import eigsh, svds

##### PCAngsd #####
# Flip signs of SVD output - Based on scikit-learn (svd_flip)
def signFlip(U, Vt):
    mcols = np.argmax(np.abs(U), axis=0)
    signs = np.sign(U[mcols, range(U.shape[1])])
    U *= signs
    Vt *= signs[:, np.newaxis]
    return U, Vt

# Estimate individual allele frequencies
def estimatePi(E, K, P):
	U, S, Vt = svds(E, k=K)
	U, Vt = signFlip(U, Vt)
	np.dot(U, S.reshape(-1,1)*Vt, out=P)
	return P, np.ascontiguousarray(Vt.T)

### PCAngsd iterations ###
def emPCA(L, f, eig, iter, tole):
	M = L.shape[0]
	N = L.shape[1]//2

	# Initiate matrices
	E = np.zeros((M, N), dtype=np.float32)
	P = np.zeros((M, N), dtype=np.float32)
	dCov = np.zeros(N)

	# Estimate covariance matrix (Fumagalli) and infer number of PCs
	if (eig == 0) or (iter == 0):
		# Prepare dosages and diagonal
		covariance_cy.covNormal(L, E, f, dCov)
		C = np.dot(E.T, E)
		np.fill_diagonal(C, dCov)
		C *= 1.0/float(M)

		if iter == 0:
			print("Returning with ngsTools covariance matrix!")
			return C, None, None, 0, False

		# Velicer's Minimum Average Partial (MAP) Test
		eVal, eVec = eigsh(C, k=min(N-1, 15)) # Eigendecomposition
		eVal = eVal[::-1] # Sorted eigenvalues
		eVal[eVal < 0] = 0
		eVec = eVec[:,::-1] # Sorted eigenvectors
		sVec = eVec*np.sqrt(eVal)
		mTest = np.empty(min(N-1, 15))

		# Loop over m-1 eigenvalues for MAP test (Shriner implementation)
		for e in np.arange(min(N-1, 15)):
			pCov = C - (np.dot(sVec[:,0:(e + 1)], sVec[:,0:(e + 1)].T))
			d = np.diag(pCov)

			if (np.sum(np.isnan(d)) > 0) or (np.sum(d == 0) > 0) or (np.sum(d < 0) > 0):
				mTest[e] = 1.0
			else:
				d = np.diagflat(1.0/np.sqrt(d))
				pr = np.dot(d, np.dot(pCov, d))
				mTest[e] = (np.sum(pr**2) - N)/(N*(N - 1))

		K = max([1, np.argmin(mTest) + 1]) # Number of principal components retained
		print(f"Using {K} principal components (MAP test).")
		del d, eVal, eVec, sVec, mTest, pCov
		if 'pr' in vars():
			del pr
	else:
		K = eig
		print(f"Using {K} principal components (manually selected).")

	# Estimate individual allele frequencies
	covariance_cy.updateNormal(L, E, f)
	P, V = estimatePi(E, K, P)
	V_pre = np.copy(V)
	print("Individual allele frequencies estimated (1).")

	# Iterative estimation
	for it in np.arange(iter):
		covariance_cy.updatePCAngsd(L, P, E, f)
		P, V = estimatePi(E, K, P)
		
		# Covergence check
		diff = shared_cy.rmse2d(V, V_pre)
		print("Individual allele frequencies estimated " + \
			f"({it+2}).\tRMSE={diff:.7f}")
		if diff < tole:
			print("PCAngsd converged.")
			converged = True
			break
		if it == (iter-1):
			print("PCAngsd did not converge!")
			converged = False
		memoryview(V_pre.ravel())[:] = memoryview(V.ravel())
	del V, V_pre

	# Estimate final covariance matrix
	dCov.fill(0.0)
	covariance_cy.covPCAngsd(L, P, E, f, dCov)
	C = np.dot(E.T, E)
	np.fill_diagonal(C, dCov)
	C *= 1.0/float(M)
	return C, P, K, it, converged
