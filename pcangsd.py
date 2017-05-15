"""
PCAngsd Framework: Population genetic analyses for NGS data using PCA.
"""

__author__ = "Jonas Meisner"

# Import functions
from helpFunctions import *
from emMAF import *
from covariance import *
from covarianceLD import *
from emInbreed import *
from emInbreedSites import *
from kinship import *
from selection import *

# Import libraries
import argparse
import numpy as np

# Argparse
parser = argparse.ArgumentParser()
parser.add_argument("input", metavar="Beagle", help="Input file of genotype likelihoods in beagle format")
parser.add_argument("-M", metavar="INT", action="store", type=int, default=100,
	help="Maximum iterations for covariance estimation (100)")
parser.add_argument("-M_tole", metavar="FLOAT", action="store", type=float, default=1e-5,
	help="Tolerance for covariance matrix estimation update (1e-5)")
parser.add_argument("-EM", metavar="INT", action="store", type=int, default=1000,
	help="Maximum iterations for population allele frequencies estimation (1000)")
parser.add_argument("-EM_tole", metavar="FLOAT", action="store", type=float, default=1e-6,
	help="Tolerance for population allele frequencies estimation update (1e-6)")
parser.add_argument("-e", metavar="INT", action="store", type=int, default=0,
	help="Manual selection of eigenvectors used for linear regression")
parser.add_argument("-reg", action="store_true",
	help="Toggle Tikhonov regularization in linear regression")
parser.add_argument("-LD", metavar="INT", action="store", type=int,
	help="Choose number of preceeding sites for LD regression")
parser.add_argument("-inbreed", metavar="INT", action="store", type=int,
	help="Compute the per-individual inbreeding coefficients by specified model")
parser.add_argument("-inbreedSites", metavar="INT", action="store", type=int,
	help="Compute the per-site inbreeding coefficients by specified model")
parser.add_argument("-inbreed_iter", metavar="INT", action="store", type=int, default=1000,
	help="Maximum iterations for inbreeding coefficients estimation (1000)")
parser.add_argument("-inbreed_tole", metavar="FLOAT", action="store", type=float, default=1e-6,
	help="Tolerance for inbreeding coefficients estimation update (1e-6)")
parser.add_argument("-selection", action="store_true",
	help="Compute a selection scan using the top principal components")
parser.add_argument("-kinship", action="store_true",
	help="Estimate the kinship matrix")
parser.add_argument("-o", metavar="Filename", action="store", help="Output file name", default="output")
parser.add_argument("-seed", metavar="INT", action="store", type=int,
	help="Choose a random seed for reproducibility")
args = parser.parse_args()


# Setting up workflow parameters
param_LD = False
param_inbreed = False
param_inbreedSites = False
param_selection = False
param_kinship = False

if args.seed != None:
	np.random.seed(args.seed)

if args.LD != None:
	param_LD = True

if args.inbreed != None:
	param_inbreed = True
	if args.inbreed == 3:
		param_kinship = True

if args.inbreedSites != None:
	param_inbreedSites = True

if args.selection:
	param_selection = True

if args.kinship:
	param_kinship = True


# Parse likelihood file
likeDF = pd.read_csv(str(args.input), sep="\t")
likeMatrix = likeDF.ix[:, 3:].as_matrix().T
pos = likeDF.ix[:, 0]
print "Parsed beagle file"


# Estimate covariance matrix
if not param_LD:
	print "\n" + "Estimating covariance matrix"
	
	# PCAngsd
	C, f, indf, nEV, mask, X = PCAngsd(likeMatrix, args.e, args.M, args.EM, args.M_tole, args.EM_tole, args.reg)

	# Reduce likelihood matrix
	likeMatrix = likeMatrix[:, mask]

	# Reset position info dataframe to evaluated sites
	pos = pos.ix[mask]
	pos.reset_index(drop=True, inplace=True)

	# Column names
	indfNames = ["ind" + str(i) for i in range(indf.shape[0])]

	# Create data frames
	covDF = pd.DataFrame(C)
	fDF = pd.DataFrame(f, columns=["maf"])
	fDF.insert(0, "marker", pos)
	indfDF = pd.DataFrame(indf.T, columns=indfNames)
	indfDF.insert(0, "marker", pos)

	# Save data frames
	covDF.to_csv(str(args.o) + ".cov", sep="\t", header=False, index=False)
	print "Saved covariance matrix as " + str(args.o) + ".cov"
	fDF.to_csv(str(args.o) + ".mafs.gz", sep="\t", index=False, compression="gzip")
	print "Saved sample allele frequencies as " + str(args.o) + ".mafs.gz"
	indfDF.to_csv(str(args.o) + ".indmafs.gz", sep="\t", index=False, compression="gzip")
	print "Saved individual allele frequencies as " + str(args.o) + ".indmafs.gz"

	# Release memory
	covDF = None
	fDF = None
	indfDF = None

	if not param_selection:
		X = None


# Estimate covariance matrix with LD regression
if param_LD:
	print "\n" + "Estimating covariance matrix with LD regression"

	# Set up positions for LD window
	posDF = pos.str.split("_", expand=True)
	posDF.ix[:, 1] = posDF.ix[:, 1].astype(int)
	posDF = posDF.as_matrix()

	# PCAngsd
	C, f, indf, nEV, mask, X = PCAngsdLD(likeMatrix, posDF, args.LD, args.e, args.M, args.EM, args.M_tole, args.EM_tole, args.reg)

	# Reduce likelihood matrix
	likeMatrix = likeMatrix[:, mask]

	# Reset position info dataframe to evaluated sites
	pos = pos.ix[mask]
	pos.reset_index(drop=True, inplace=True)

	# Column names
	indfNames = ["ind" + str(i) for i in range(indf.shape[0])]

	# Create data frames
	covDF = pd.DataFrame(C)
	fDF = pd.DataFrame(f, columns=["maf"])
	fDF.insert(0, "marker", pos)
	indfDF = pd.DataFrame(indf.T, columns=indfNames)
	indfDF.insert(0, "marker", pos)

	# Save data frames
	covDF.to_csv(str(args.o) + ".cov", sep="\t", header=False, index=False)
	print "Saved covariance matrix as " + str(args.o) + ".cov"
	fDF.to_csv(str(args.o) + ".mafs.gz", sep="\t", index=False, compression="gzip")
	print "Saved sample allele frequencies as " + str(args.o) + ".mafs.gz"
	indfDF.to_csv(str(args.o) + ".indmafs.gz", sep="\t", index=False, compression="gzip")
	print "Saved individual allele frequencies as " + str(args.o) + ".indmafs.gz"

	# Release memory
	posDF = None
	covDF = None
	fDF = None
	indfDF = None

	if not param_selection:
		X = None


# Selection scan
if param_selection:
	print "\n" + "Performing selection scan"

	# Perform selection scan and save data frame
	pcNames = ["PC" + str(pc) for pc in range(1, nEV+1)]
	chisqDF = pd.DataFrame(selectionScan(X, C, nEV).T, columns=pcNames)
	chisqDF.insert(0, "marker", pos)
	chisqDF.to_csv(str(args.o) + ".selection.gz", sep="\t", index=False, compression="gzip")
	print "Saved selection statistics for the top PCs as " + str(args.o) + ".selection.gz"

	# Release memory
	chisqDF = None


# Kinship estimation
if param_kinship:
	print "\n" + "Estimating kinship matrix"

	# Perform kinship estimation
	phi = kinshipConomos(likeMatrix, indf)

	# Save data frame
	phiDF = pd.DataFrame(phi)
	phiDF.to_csv(str(args.o) + ".kinship", sep="\t", header=False, index=False)

	# Release memory
	phiDF = None


# Individual inbreeding coefficients
if param_inbreed and args.inbreed == 1:
	print "\n" + "Estimating inbreeding coefficients using maximum likelihood estimator (EM)"

	# Estimating inbreeding coefficients
	F = inbreedEM(likeMatrix, indf, 1, args.inbreed_iter, args.inbreed_tole)

	# Save data frame
	cNames = ["ind" + str(ind) for ind in range(F.shape[0])]
	F_DF = pd.DataFrame(F)
	F_DF.insert(0, "indID", cNames)
	F_DF.to_csv(str(args.o) + ".inbreed.gz", sep="\t", header=False, index=False)

	# Release memory
	F = None
	F_DF = None

elif param_inbreed and args.inbreed == 2:
	print "\n" + "Estimating inbreeding coefficients using simple estimator (EM)"

	# Estimating inbreeding coefficients
	F = inbreedEM(likeMatrix, indf, 2, args.inbreed_iter, args.inbreed_tole)

	# Save data frame
	cNames = ["ind" + str(ind) for ind in range(F.shape[0])]
	F_DF = pd.DataFrame(F)
	F_DF.insert(0, "indID", cNames)
	F_DF.to_csv(str(args.o) + ".inbreed.gz", sep="\t", header=False, index=False)

	# Release memory
	F = None
	F_DF = None
	
elif param_inbreed and param_kinship:
	print "\n" + "Estimating inbreeding coefficients using kinship estimator (Conomos et al.)"

	# Estimating inbreeding coefficients by previously estimated kinship matrix
	F = 2*phi.diagonal() - 1

	# Save data frame
	cNames = ["ind" + str(ind) for ind in range(F.shape[0])]
	F_DF = pd.DataFrame(F)
	F_DF.insert(0, "indID", cNames)
	F_DF.to_csv(str(args.o) + ".inbreed.gz", sep="\t", header=False, index=False)

	# Release memory
	F = None
	phi = None
	F_DF = None


# Per-site inbreeding coefficients
if param_inbreedSites and args.inbreedSites == 1:
	print "\n" + "Estimating per-site inbreeding coefficients using maximum likelioohd estimator (EM)"

	# Estimating per-site inbreeding coefficients
	F = inbreedSitesEM(likeMatrix, indf, 1, args.inbreed_iter, args.inbreed_tole)

	# Save data frame
	F_DF = pd.DataFrame(F, columns=["F"])
	F_DF.insert(0, "marker", pos)
	F_DF.to_csv(str(args.o) + ".inbreedSites.gz", sep="\t", index=False, compression="gzip")

	# Release memory
	F = None
	F_DF = None

elif param_inbreedSites and args.inbreedSites == 2:
	print "\n" + "Estimating per-site inbreeding coefficients using simple estimator (EM)"

	# Estimating per-site inbreeding coefficients
	F = inbreedSitesEM(likeMatrix, indf, 2, args.inbreed_iter, args.inbreed_tole)

	# Save data frame
	F_DF = pd.DataFrame(F, columns=["F"])
	F_DF.insert(0, "marker", pos)
	F_DF.to_csv(str(args.o) + ".inbreedSites.gz", sep="\t", index=False, compression="gzip")

	# Release memory
	F = None
	F_DF = None