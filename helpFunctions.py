"""
Help functions to use in the PCAngsd framework.
"""

__author__ = "Jonas Meisner"

# Import libraries
import numpy as np
import pandas as pd

# Root mean squared error
def rmse(m1, m2):
	return np.sqrt(np.mean((m1 - m2)**2))


# Compute bias
def bias(mEst, mTrue):
	return np.mean(mEst-mTrue)

