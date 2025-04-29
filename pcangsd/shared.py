import numpy as np
from math import ceil
from pcangsd import reader_cy
from pcangsd import shared_cy

##### Functions #####
# Read PLINK files
def readPlink(bfile, e):
	# Find length of fam-file
	N = 0
	with open(f"{bfile}.fam", "r") as fam:
		for _ in fam:
			N += 1
	N_bytes = ceil(N/4) # Length of bytes to describe N individuals

	# Read .bed file
	with open(f"{bfile}.bed", "rb") as bed:
		G = np.fromfile(bed, dtype=np.uint8, offset=3)
	assert (G.shape[0] % N_bytes) == 0, "bim file doesn't match!"
	M = G.shape[0]//N_bytes
	G.shape = (M, N_bytes)

	# Convert genotypes into genotype likelihood
	L = np.zeros((M, 2*N), dtype=np.float32)
	reader_cy.convertBed(L, G, e)
	return L, M, N

# Estimate MAF
def emMAF(L, iter, tole):
	M = L.shape[0]
	f = np.full(M, 0.25)
	shared_cy.emMAF_update(L, f)

	# QN containers
	f0 = np.copy(f)
	f1 = np.zeros(M)
	f2 = np.zeros(M)

	# EM algorithm
	for it in np.arange(iter):
		memoryview(f0)[:] = memoryview(f)
		shared_cy.emMAF_accel(L, f, f1)
		shared_cy.emMAF_accel(L, f1, f2)
		
		# Add safety check for no missingness
		if it == 0:
			if np.allclose(f, f1) or np.allclose(f1, f2):
				print("EM (MAF) converged. No missingness!")
				memoryview(f)[:] = memoryview(f2)
				break
		
		# Accelerated jump
		shared_cy.emMAF_alpha(f, f1, f2)

		# Stabilization step and convergence check
		shared_cy.emMAF_update(L, f)
		diff = shared_cy.rmse1d(f, f0)
		if diff < tole:
			print(f"EM (MAF) converged at iteration: {it+1}")
			break
		if it == (iter-1):
			print("EM (MAF) failed to converge!")
	return f

# Genotype posteriors
def estimatePost(L, P, F):
	M, N = P.shape
	G = np.zeros((M, 3*N), dtype=np.float32)

	# Estimate genotype posteriors based on individual allele frequencies
	if F is None:
		shared_cy.post(L, P, G)
	else:
		shared_cy.postInbreed(L, P, G, F)
	return G

# Genotype calling
def callGeno(L, P, F, delta):
	M, N = P.shape
	G = np.zeros((M, N), dtype=np.int8)

	# Call genotypes with highest posterior probabilities
	if F is None:
		shared_cy.geno(L, P, G, delta)
	else:
		shared_cy.genoInbreed(L, P, F, G, delta)
	return G

# Fake frequencies for educational purposes
def fakeFreqs(f, M, N):
	P = np.zeros((M, N), dtype=np.float32)
	shared_cy.freqs(P, f)
	return P
