"""
PCAngsd.
Estimate per-site and per-individual inbreeding coefficients.
"""

__author__ = "Jonas Meisner"

# Libraries
import numpy as np
from math import isclose, sqrt

# Import scripts
from pcangsd import shared_cy
from pcangsd import inbreed_cy

##### Inbreeding coefficients #####
### Per-site inbreeding coefficients ###
def inbreedSites(L, P, iter, tole, t):
	m, _ = P.shape
	F = np.zeros(m)
	T = np.zeros(m)

	# SqS3 containers
	F1 = np.zeros(m)
	F2 = np.zeros(m)
	d1 = np.zeros(m)
	d2 = np.zeros(m)
	d3 = np.zeros(m)

	# EM algorithm
	for it in range(iter):
		F0 = np.copy(F)

		# 1st step
		inbreed_cy.inbreedSites_accel(L, P, F, F1, d1, t)
		sr2 = shared_cy.vecSumSquare(d1)

		# 2nd step
		inbreed_cy.inbreedSites_accel(L, P, F1, F2, d2, t)
		shared_cy.vecMinus(d2, d1, d3)
		sv2 = shared_cy.vecSumSquare(d3)

		# Safety break due to zero missingness
		if isclose(sv2, 0.0):
			F = np.copy(F2)
			print(f"Inbreeding coefficients estimated ({it+1}).\tRMSE=0.0")
			print("EM (inbreeding - sites) converged.")
			break

		# Alpha step
		alpha = -max(1.0, sqrt(sr2/sv2))
		shared_cy.vecUpdate(F, F0, d1, d3, alpha)

		# Stabilization step and convergence check
		inbreed_cy.inbreedSites_update(L, P, F, t)
		diff = shared_cy.rmse1d(F, F0)
		print(f"Inbreeding coefficients estimated ({it+1}).\tRMSE={np.round(diff,9)}")
		if diff < tole:
			print("EM (inbreeding - sites) converged.")
			break
		if it == (iter - 1):
			print("EM (inbreeding - sites) did not converge!")

	# LRT statistic
	inbreed_cy.loglike(L, P, F, T, t)
	del F0, F1, F2, d1, d2, d3
	return F, T

### Per-individual inbreeding coefficients ###
def inbreedSamples(L, P, iter, tole, t):
	_, n = P.shape
	F = np.zeros(n)
	Ftmp = np.zeros(n)
	Etmp = np.zeros(n)
	
	# SqS3 containers
	F1 = np.zeros(n)
	F2 = np.zeros(n)
	d1 = np.zeros(n)
	d2 = np.zeros(n)
	d3 = np.zeros(n)

	# EM algorithm
	for it in range(iter):
		F0 = np.copy(F)

		# 1st step
		inbreed_cy.inbreedSamples_accel(L, P, F, F1, d1, Ftmp, Etmp, t)
		sr2 = shared_cy.vecSumSquare(d1)

		# 2nd step
		inbreed_cy.inbreedSamples_accel(L, P, F1, F2, d2, Ftmp, Etmp, t)
		shared_cy.vecMinus(d2, d1, d3)
		sv2 = shared_cy.vecSumSquare(d3)
		
		# Safety break due to zero missingness
		if isclose(sv2, 0.0):
			F = np.copy(F2)
			print(f"Inbreeding coefficients estimated ({it+1}).\tRMSE=0.0")
			print("EM (inbreeding - samples) converged.")
			break

		# Alpha step
		alpha = -max(1.0, sqrt(sr2/sv2))
		shared_cy.vecUpdate(F, F0, d1, d3, alpha)

		# Stabilization step and convergence check
		inbreed_cy.inbreedSamples_update(L, P, F, Ftmp, Etmp, t)
		diff = shared_cy.rmse1d(F, F0)
		print(f"Inbreeding coefficients estimated ({it+1}).\tRMSE={np.round(diff,9)}")
		if diff < tole:
			print("EM (inbreeding - samples) converged.")
			break
		if it == (iter - 1):
			print("EM (inbreeding - samples) did not converge!")
	del Ftmp, Etmp, F0, F1, F2, d1, d2, d3
	return F
