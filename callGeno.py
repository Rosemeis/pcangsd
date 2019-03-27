"""
Call genotypes from posterior genotype probabilities using estimated individual allele frequencies as prior.
Can be performed with and without taking inbreeding into account.
"""

__author__ = "Jonas Meisner"

# Import libraries
import numpy as np
import callGeno_cy

##### Genotype calling #####
def callGeno(L, Pi, F, delta, t):
	n, m = Pi.shape
	G = np.empty((n, m), dtype=np.int8)

	# Call genotypes with highest posterior probabilities
	if F is None:
		callGeno_cy.geno(L, Pi, delta, G, t)
	else:
		callGeno_cy.genoInbreed(L, Pi, F, delta, G, t)

	return G