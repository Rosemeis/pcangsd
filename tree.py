"""
Tree-construction methods
"""

__author__ = "Jonas Meisner"

# Import libraries
import numpy as np
import tree_cy

# Construct covariance from Pi
def covarPi(Pi, f, t):
	n, m = Pi.shape
	PiNorm = np.zeros((n, m), dtype=np.float32)

	# Call functions
	tree_cy.standardizePi(Pi, f, PiNorm, t) # Standardize IAF
	Covar = np.dot(PiNorm, PiNorm.T)/m # IAF covariance mat
	del PiNorm
	return Covar.astype(np.float32)

# Create neighbour-joining tree
def constructTree(Covar):
	n = Covar.shape[0]
	D0 = np.zeros((n, n), dtype=np.float32)
	indIndex = list(range(n))

	# Distance matrix and Q matrix
	tree_cy.estimateDist(Covar, D0)
	Dsum = np.sum(D0, axis=1)

	n1 = n - 1
	for i in range(n-3):
		# Create new node
		Q = np.zeros(D0.shape, dtype=np.float32)
		tree_cy.estimateQ(D0, Q, Dsum)
		np.fill_diagonal(Q, np.inf)
		pairA, pairB = sorted(np.unravel_index(np.argmin(Q), Q.shape))
		del Q
		i2 = indIndex.pop(pairB)
		i1 = indIndex.pop(pairA)
		dist1 = max(0, 0.5*D0[pairA, pairB] + (1.0/(2.0*(n1 + 1 - 2)))*(Dsum[pairA] - Dsum[pairB]))
		dist2 = max(0, D0[pairA, pairB] - dist1)
		indIndex.append("(%s:%f, %s:%f)" % (str(i1), dist1, str(i2), dist2))

		# Create new distance matrix
		D = np.zeros((n1, n1), dtype=np.float32)
		tree_cy.updateD(D0, D, pairA, pairB) # Update new distances
		D0 = np.copy(D)
		del D
		Dsum = np.sum(D0, axis=1)
		n1 -= 1
	# Last joining (n=3)
	Q = np.zeros(D0.shape, dtype=np.float32)
	tree_cy.estimateQ(D0, Q, Dsum)
	np.fill_diagonal(Q, np.inf)
	pairA, pairB = np.unravel_index(np.argmin(Q), Q.shape)
	del Q
	lastI = list(set([0,1,2]) ^ set([pairA, pairB]))[0]
	i1 = indIndex[pairA]
	i2 = indIndex[pairB]
	i3 = indIndex[lastI]
	dist1 = max(0, 0.5*D0[pairA, pairB] + (1.0/(2.0*(n1 + 1 - 2)))*(Dsum[pairA] - Dsum[pairB]))
	dist2 = max(0, D0[pairA, pairB] - dist1)
	dist3 = max(0, 0.5*(D0[pairA, lastI] + D0[pairB, lastI] - D0[pairA, pairB]))
	return "(%s:%f, %s:%f, %s:%f)" % (str(i1), dist1, str(i3), dist3, str(i2), dist2)
