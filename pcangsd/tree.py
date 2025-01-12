import numpy as np
from pcangsd import tree_cy

##### Tree estimation #####
# Construct covariance from normalized P
def covariancePi(P, f):
	M, N = P.shape
	Pi = np.zeros((M, N), dtype=np.float32)

	# Call functions
	tree_cy.standardizePi(P, Pi, f)
	C = np.dot(Pi.T, Pi)*(1.0/float(M))
	return C

# Create neighbour-joining tree
def constructTree(C, sList):
	n = C.shape[0]
	D0 = np.zeros((n, n), dtype=np.float32)
	indIndex = sList.copy()

	# Distance matrix and Q matrix
	tree_cy.estimateD(C, D0)
	D_sum = np.sum(D0, axis=1)
	n1 = n - 1
	for _ in np.arange(n-3):
		# Create new node
		Q = np.zeros_like(D0)
		tree_cy.estimateQ(D0, Q, D_sum)
		np.fill_diagonal(Q, np.inf)
		pA, pB = sorted(np.unravel_index(np.argmin(Q), Q.shape))
		del Q
		i2 = indIndex.pop(pB)
		i1 = indIndex.pop(pA)
		dist1 = max(0, 0.5*D0[pA, pB] + (1.0/(2.0*(n1 + 1 - 2)))* \
			(D_sum[pA] - D_sum[pB]))
		dist2 = max(0, D0[pA, pB] - dist1)
		indIndex.append(f"({i1}:{dist1}, {i2}:{dist2})")

		# Create new distance matrix
		D = np.zeros((n1, n1), dtype=np.float32)
		tree_cy.updateD(D0, D, pA, pB) # Update new distances
		D0 = np.copy(D)
		del D
		D_sum = np.sum(D0, axis=1)
		n1 -= 1
	
	# Last joining (n=3)
	Q = np.zeros_like(D0)
	tree_cy.estimateQ(D0, Q, D_sum)
	np.fill_diagonal(Q, np.inf)
	pA, pB = np.unravel_index(np.argmin(Q), Q.shape)
	del Q
	iN = list(set([0,1,2])^set([pA, pB]))[0]
	i1 = indIndex[pA]
	i2 = indIndex[pB]
	i3 = indIndex[iN]
	dist1 = max(0, 0.5*D0[pA, pB] + (1.0/(2.0*(n1 + 1 - 2)))* \
		(D_sum[pA] - D_sum[pB]))
	dist2 = max(0, D0[pA, pB] - dist1)
	dist3 = max(0, 0.5*(D0[pA, iN] + D0[pB, iN] - D0[pA, pB]))
	return f"({i1}:{dist1}, {i3}:{dist3}, {i2}:{dist2})"
