"""
EM algorithms to estimate per-site or per-individual inbreeding coefficients for NGS data using genotype likelihoods
and pre-computed allele frequencies (both sample average or individual)..
"""

__author__ = "Jonas Meisner"

# Import libraries
import numpy as np
import inbreed_cy
import shared

# EM algorithm for estimation of per-site inbreeding coefficients
def inbreedSitesEM(L, Pi, m_iter, m_tole, t):
	n, m = Pi.shape
	F = np.zeros(m, dtype=np.float32)
	F_prev = np.copy(F)

	# EM algorithm
	for iteration in range(1, m_iter+1):
		inbreed_cy.emInbreedSites_update(L, Pi, F, t)

		# Break EM update if converged
		updateDiff = shared.rmse1d(F, F_prev)
		print("Inbreeding coefficients estimated (" + str(iteration) + "). RMSE=" + str(updateDiff))
		if updateDiff < m_tole:
			print("EM (Inbreeding - sites) converged at iteration: " + str(iteration))
			break
		F_prev = np.copy(F)

	# LRT test statistic
	lrt = np.empty(m, dtype=np.float32)
	inbreed_cy.loglike(L, Pi, F, lrt, t)
	return F, lrt

# EM algorithm for estimation of per-individual inbreeding coefficients
def inbreedEM(L, Pi, model, m_iter, m_tole, t):
	n, m = Pi.shape
	
	# Model initialization
	if model == 1:
		print("Using Simple model")
		F = np.zeros(n, dtype=np.float32)
	elif model == 2:
		print("Using Hall model")
		F = np.ones(n, dtype=np.float32)*0.1
	F_prev = np.copy(F)

	for iteration in range(1, m_iter+1):
		if model == 1:
			inbreed_cy.emInbreed_update(L, Pi, F, t)
		elif model == 2:
			inbreed_cy.emHall_update(L, Pi, F, t)

		# Break EM update if converged
		updateDiff = shared.rmse1d(F, F_prev)
		print("Inbreeding coefficients estimated (" + str(iteration) + "). RMSE=" + str(updateDiff))
		if updateDiff < m_tole:
			print("EM (Inbreeding - individuals) converged at iteration: " + str(iteration))
			break
		F_prev = np.copy(F)

	return F