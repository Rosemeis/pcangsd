"""
PCAngsd.
Estimate per-site and per-individual inbreeding coefficients.
"""

__author__ = "Jonas Meisner"

# Libraries
import numpy as np

# Import scripts
from pcangsd import shared_cy
from pcangsd import inbreed_cy

##### Inbreeding coefficients #####
### Per-site inbreeding coefficients ###
def inbreedSites(L, P, iter, tole, t):
	m, _ = P.shape
	F = np.zeros(m, dtype=np.float32)
	T = np.zeros(m, dtype=np.float64)

	# EM algorithm
	for i in range(iter):
		F_prev = np.copy(F)
		inbreed_cy.inbreedSites_update(L, P, F, t)

		# Check for convergence
		diff = shared_cy.rmse1d(F, F_prev)
		print("Inbreeding coefficients estimated (" + str(i+1) + \
			"). RMSE=" + str(diff))
		if diff < tole:
			print("Converged.")
			break

	# LRT statistic
	inbreed_cy.loglike(L, P, F, T, t)
	del F_prev
	return F, T

### Per-individual inbreeding coefficients ###
def inbreedSamples(L, P, iter, tole, t):
	_, n = P.shape
	F = np.zeros(n, dtype=np.float32)
	Ftmp = np.zeros(n, dtype=np.float32)
	Etmp = np.zeros(n, dtype=np.float32)

	# EM algorithm
	for i in range(iter):
		F_prev = np.copy(F)
		inbreed_cy.inbreedSamples_update(L, P, F, Ftmp, Etmp, t)

		# Check for convergence
		diff = shared_cy.rmse1d(F, F_prev)
		print("Inbreeding coefficients estimated (" + str(i+1) + \
			"). RMSE=" + str(diff))
		if diff < tole:
			print("Converged.")
			break
	del F_prev
	return F
