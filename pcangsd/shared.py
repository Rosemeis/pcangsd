"""
PCAngsd.
Functions used in various analyses.
"""

__author__ = "Jonas Meisner"

# Libraries
import subprocess
import numpy as np
from math import isclose, sqrt

# Import scripts
from pcangsd import shared_cy

##### Functions #####
# Find length of PLINK files
def extract_length(filename):
	process = subprocess.Popen(['wc', '-l', filename], stdout=subprocess.PIPE)
	result, err = process.communicate()
	return int(result.split()[0])


# Estimate MAF
def emMAF(L, iter, tole, t):
	m = L.shape[0] # Number of sites
	f = np.full(m, 0.25, dtype=np.float32)

	# SqS3 containers
	f1 = np.zeros(m, dtype=np.float32)
	f2 = np.zeros(m, dtype=np.float32)
	d1 = np.zeros(m, dtype=np.float32)
	d2 = np.zeros(m, dtype=np.float32)
	d3 = np.zeros(m, dtype=np.float32)

	# EM algorithm
	for it in range(iter):
		f0 = np.copy(f)

		# 1st step
		shared_cy.emMAF_accel(L, f, f1, d1, t)
		sr2 = shared_cy.vecSumSquare(d1)

		# 2nd step
		shared_cy.emMAF_accel(L, f1, f2, d2, t)
		shared_cy.vecMinus(d2, d1, d3)
		sv2 = shared_cy.vecSumSquare(d3)

		# Safety break due to zero missingness
		if isclose(sv2, 0.0):
			f = np.copy(f2)
			print(f"EM (MAF) converged at iteration: {it+1}")
			break

		# Alpha step
		alpha = -max(1.0, sqrt(sr2/sv2))
		shared_cy.vecUpdate(f, f0, d1, d3, alpha)

		# Stabilization step and convergence check
		shared_cy.emMAF_update(L, f, t)
		diff = shared_cy.rmse1d(f, f0)
		if diff < tole:
			print(f"EM (MAF) converged at iteration: {it+1}")
			break
		if it == (iter - 1):
			print("EM (MAF) failed to converge!")
	del f1, f2, d1, d2, d3
	return f


# Genotype calling
def callGeno(L, P, F, delta, t):
	m, n = P.shape
	G = np.zeros((m, n), dtype=np.int8)

	# Call genotypes with highest posterior probabilities
	if F is None:
		shared_cy.geno(L, P, G, delta, t)
	else:
		shared_cy.genoInbreed(L, P, F, G, delta, t)
	return G


# Fake frequencies for educational purposes
def fakeFreqs(f, m, n, t):
	P = np.zeros((m, n), dtype=np.float32)

	# Fill up matrix
	shared_cy.freqs(P, f, t)
	return P
