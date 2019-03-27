"""
Kinship estimator using genotype likelihoods based on PC-Relate.
"""

__author__ = "Jonas Meisner"

# Import libraries
import numpy as np
import kinship_cy
import covariance_cy

# Estimate kinship matrix
def kinshipConomos(L, Pi, t):
	n, m = Pi.shape

	# Initiate containers
	E = np.empty((n, m), dtype=np.float32)
	dKin = np.empty(n, dtype=np.float32)
	temp1 = np.empty((n, n), dtype=np.float32)
	temp2 = np.empty((n, n), dtype=np.float32)

	# Dosages
	covariance_cy.updatePCAngsd(L, Pi, E, t)

	# Kinship computations
	kinship_cy.diagKinship(L, Pi, dKin, t)
	kinship_cy.numeratorKin(E, Pi, t)
	np.dot(E, E.T, out=temp1)
	kinship_cy.denominatorKin(E, Pi, t)
	np.dot(E, E.T, out=temp2)
	temp2 *= 4
	temp1 /= temp2
	np.fill_diagonal(temp1, dKin) # Insert correct diagonal
	return temp1