import numpy as np
from pcangsd import shared_cy
from pcangsd import inbreed_cy

##### Inbreeding coefficients #####
### Per-site inbreeding coefficients ###
def inbreedSites(L, P, iter, tole):
	M, _ = P.shape
	F = np.zeros(M)
	inbreed_cy.inbreedSites_update(L, P, F)

	# QN containers
	F0 = np.zeros(M)
	F1 = np.zeros(M)
	F2 = np.zeros(M)

	# EM algorithm
	for it in np.arange(iter):
		memoryview(F0)[:] = memoryview(F)
		inbreed_cy.inbreedSites_accel(L, P, F, F1)
		inbreed_cy.inbreedSites_accel(L, P, F1, F2)
		shared_cy.inbreed_alpha(F, F1, F2)

		# Stabilization step and convergence check
		inbreed_cy.inbreedSites_update(L, P, F)
		diff = shared_cy.rmse1d(F, F0)
		print(f"Inbreeding coefficients estimated ({it+1}).\tRMSE={diff:.7f}")
		if diff < tole:
			print("EM (inbreeding - sites) converged.")
			break
		if it == (iter-1):
			print("EM (inbreeding - sites) did not converge!")
	del F0, F1, F2

	# LRT statistic
	T = np.zeros(M)
	inbreed_cy.loglike(L, P, F, T)
	return F, T

### Per-sample inbreeding coefficients ###
def inbreedSamples(L, P, iter, tole):
	_, N = P.shape
	F = np.zeros(N)
	Ftmp = np.zeros(N)
	Etmp = np.zeros(N)
	inbreed_cy.inbreedSamples_update(L, P, F, Ftmp, Etmp)
	
	# QN containers
	F0 = np.zeros(N)
	F1 = np.zeros(N)
	F2 = np.zeros(N)

	# EM algorithm
	for it in np.arange(iter):
		memoryview(F0)[:] = memoryview(F)
		inbreed_cy.inbreedSamples_accel(L, P, F, F1, Ftmp, Etmp)
		inbreed_cy.inbreedSamples_accel(L, P, F1, F2, Ftmp, Etmp)
		shared_cy.inbreed_alpha(F, F1, F2)

		# Stabilization step and convergence check
		inbreed_cy.inbreedSamples_update(L, P, F, Ftmp, Etmp)
		diff = shared_cy.rmse1d(F, F0)
		print(f"Inbreeding coefficients estimated ({it+1}).\tRMSE={diff:.7f}")
		if diff < tole:
			print("EM (inbreeding - samples) converged.")
			break
		if it == (iter-1):
			print("EM (inbreeding - samples) did not converge!")
	return F
