"""
EM algorithm to estimate the per-site population allele frequencies for NGS data using genotype likelihoods.
Maximum likelihood estimator.
"""

__author__ = "Jonas Meisner"

# Import libraries
import numpy as np
import shared

##### Functions #####
# EM algorithm for estimation of population allele frequencies
def alleleEM(L, m_iter, m_tole, t):
	_, m = L.shape # Dimension of likelihood matrix
	f = np.ones(m, dtype=np.float32)*0.25 # Uniform initialization
	newF = np.empty(m, dtype=np.float32) # Helper vector

	for iteration in range(1, m_iter+1): # EM iterations
		shared.emMaf_update(L, f, newF, t) # Updated allele frequencies

		# Break EM update if converged
		if iteration > 1:
			diff = shared.rmse1d(f, f_prev)
			if diff < m_tole:
				print("EM (MAF) converged at iteration: " + str(iteration) + "\n")
				break
		f_prev = np.copy(f)
	return f