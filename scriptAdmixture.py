"""
Script for running multiple admixture estimations based on PCAngsd.
"""

__author__ = "Jonas Meisner"

# Import libraries
import argparse
import numpy as np
import pandas as pd

# Import functions
from admixture import *

# Argparse
parser = argparse.ArgumentParser()
parser.add_argument("-freqs", metavar="FILE", 
	help="Input file of individual allele frequencies from PCAngsd (.indmafs.gz)")
parser.add_argument("-cov", metavar="FILE",
	help="Input file of covariance matrix (.cov)")
parser.add_argument("-K", metavar="INT-LIST", nargs="+", type=int,
	help="Integer list of number of ancestral populations to use in admixture estimation")
parser.add_argument("-admix_alpha", metavar="FLOAT-LIST", nargs="+", type=float,
	help="Floating point list of different sparseness factors")
parser.add_argument("-admix_seed", metavar="INT-LIST", nargs="+", type=int,
	help="Seed list for different K-means")
parser.add_argument("-admix_iter", metavar="INT", type=int, default=100,
	help="Maximum iterations for admixture proportions estimation - NMF (100)")
parser.add_argument("-admix_tole", metavar="FLOAT", type=float, default=1e-5,
	help="Tolerance for admixture proportions estimation update - EM (1e-5)")
parser.add_argument("-admix_batch", metavar="INT", type=int, default=50,
	help="Number of batches used for stochastic gradient descent (50)")
parser.add_argument("-o", metavar="OUTPUT", help="Prefix output file name", default="pcangsd.admixture")
args = parser.parse_args()

print "Parsing individual allele frequencies"
indf = (pd.read_csv(str(args.freqs), sep="\t", header=None, dtype=np.float32, engine="c").as_matrix()).T

print "Parsing covariance matrix"
C = pd.read_csv(str(args.cov), sep="\t", header=None).as_matrix()

for k in args.K:
	for a in args.admix_alpha:
		for s in args.admix_seed:
			print "\n" + "Estimating admixture with K=" + str(k) + ", alpha=" + str(a) + " and seed=" + str(s)
			Q_admix, F_admix = admixNMF(indf, k, C, a, args.admix_iter, args.admix_tole, s, args.admix_batch)

			# Save data frame
			pd.DataFrame(Q_admix).to_csv(str(args.o) + ".K" + str(k) + ".a" + str(a) + ".s" + str(s) + ".qopt", sep="\t", header=False, index=False)
			print "Saved admixture proportions as " + str(args.o) + ".K" + str(k) + ".a" + str(a) + ".s" + str(s) + ".qopt"

			pd.DataFrame(F_admix).to_csv(str(args.o) + ".K" + str(k) + ".a" + str(a) + ".fopt.gz", sep="\t", header=False, index=False, compression="gzip")
			print "Saved population-specific allele frequencies as " + str(args.o) + ".K" + str(k) + ".a" + str(a) + ".s" + str(s) + ".fopt.gz"