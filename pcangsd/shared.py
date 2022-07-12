"""
PCAngsd.
Functions used in various analyses.
"""

__author__ = "Jonas Meisner"

# Libraries
import numpy as np

# Import scripts
from pcangsd import shared_cy

##### Functions #####
### Estimate MAF ###
def emMAF(L, iter, tole, t):
    m = L.shape[0] # Number of sites
    f = np.empty(m, dtype=np.float32)
    f.fill(0.25) # Uniform initialization
    f_prev = np.copy(f)
    for i in range(iter):
        shared_cy.emMAF_update(L, f, t)
        diff = shared_cy.rmse1d(f, f_prev)
        if diff < tole:
            print("EM (MAF) converged at iteration: " + str(i+1))
            break
        f_prev = np.copy(f)
    return f


### Genotype calling ###
def callGeno(L, P, F, delta, t):
    m, n = P.shape
    G = np.zeros((m, n), dtype=np.int8)

    # Call genotypes with highest posterior probabilities
    if F is None:
        shared_cy.geno(L, P, G, delta, t)
    else:
        shared_cy.genoInbreed(L, P, F, G, delta, t)
    return G


### Genotype calling ###
def calcPost(L, P, F, t):
    m, n = P.shape
    G = np.zeros((n, 3*n), dtype=dtype=np.float32) # in this function, G holds the genotype posteriors 

    # Calculate genotype posteriors
    if F is None:
        shared_cy.gpost(L, P, G, t)
    else:
        #shared_cy.genoInbreed(L, P, F, G, delta, t)
        raise Exception("Calling posteriors with inbreeding not yet implemented.")
    return G

