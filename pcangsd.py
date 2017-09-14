"""
PCAngsd Framework: Population genetic analyses for NGS data using PCA. Main caller.
"""

__author__ = "Jonas Meisner"

# Import functions
from helpFunctions import *
from helpChunks import *
from emMAF import *
from covariance import *
from callGeno import *
from emInbreed import *
from emInbreedSites import *
from kinship import *
from selection import *

# Import libraries
import argparse
import numpy as np
import pandas as pd

# Argparse
parser = argparse.ArgumentParser(prog="PCAngsd")
parser.add_argument("--version", action="version", version="%(prog)s 0.3")
parser.add_argument("-beagle", metavar="FILE", 
	help="Input file of genotype likelihoods in Beagle format")
parser.add_argument("-beaglelist", metavar="LIST", 
	help="List of input files of genotype likelihoods in Beagle format")
parser.add_argument("-M", metavar="INT", type=int, default=100,
	help="Maximum iterations for covariance estimation (100)")
parser.add_argument("-M_tole", metavar="FLOAT", type=float, default=1e-4,
	help="Tolerance for covariance matrix estimation update (1e-4)")
parser.add_argument("-EM", metavar="INT", type=int, default=200,
	help="Maximum iterations for population allele frequencies estimation (200)")
parser.add_argument("-EM_tole", metavar="FLOAT", type=float, default=1e-4,
	help="Tolerance for population allele frequencies estimation update (1e-4)")
parser.add_argument("-e", metavar="INT", type=int, default=0,
	help="Manual selection of eigenvectors used for linear regression")
parser.add_argument("-reg", metavar="FLOAT", type=float, default=0,
	help="(PROTOTYPE) Perform ridge regression in estimation of individual allele frequencies")
parser.add_argument("-scaled", action="store_true",
	help="(PROTOTYPE) Perform scaled regression based on eigenvalues")
parser.add_argument("-LD", metavar="INT", type=int, default=0,
	help="(PROTOTYPE) Perform LD regression on a specified number of preceding sites")
parser.add_argument("-geno", metavar="FLOAT", type=float,
	help="Call genotypes from posterior probabilities using individual allele frequencies as prior")
parser.add_argument("-genoInbreed", metavar="FLOAT", type=float,
	help="Call genotypes from posterior probabilities using individual allele frequencies and inbreeding coefficients as prior")
parser.add_argument("-inbreed", metavar="INT", type=int,
	help="Compute the per-individual inbreeding coefficients by specified model")
parser.add_argument("-inbreedSites", action="store_true",
	help="Compute the per-site inbreeding coefficients by specified model and LRT")
parser.add_argument("-inbreed_iter", metavar="INT", type=int, default=200,
	help="Maximum iterations for inbreeding coefficients estimation (200)")
parser.add_argument("-inbreed_tole", metavar="FLOAT", type=float, default=1e-4,
	help="Tolerance for inbreeding coefficients estimation update (1e-4)")
parser.add_argument("-selection", metavar="INT", type=int,
	help="Compute a selection scan using the top principal components by specified model")
parser.add_argument("-kinship", action="store_true",
	help="Estimate the kinship matrix")
parser.add_argument("-chunksize", metavar="INT", type=int, default=0,
	help="Enable chunk-mode for estimations of site-parameters. Specify chunksize")
parser.add_argument("-cov", metavar="FILE",
	help="Pre-estimated covariance matrix")
parser.add_argument("-F", metavar="FILE",
	help="Pre-estimated per-individual inbreeding coefficients")
parser.add_argument("-o", metavar="OUTPUT", help="Output file name", default="pcangsd")
args = parser.parse_args()

print "Running PCAngsd 0.3"
assert (args.beagle != None or args.beaglelist != None), "Missing Beagle file(s)!"

# Setting up workflow parameters
param_call = False
param_inbreed = False
param_inbreedSites = False
param_selection = False
param_kinship = False

if args.inbreed != None:
	param_inbreed = True
	if args.inbreed == 3:
		param_kinship = True

if args.inbreedSites:
	param_inbreedSites = True

if args.selection != None:
	param_selection = True

if args.kinship:
	param_kinship = True

if args.geno != None:
	param_call = True

if args.genoInbreed:
	assert param_inbreed, "Inbreeding coefficients must be estimated in order to use -genoInbreed! Use -inbreed parameter!"

# Read in pre-estimated parameters if chunk-mode has been selected
if args.chunksize > 0:
	assert args.beagle != None, "Chunk-mode only supports estimations on a single Beagle file!"
	assert args.cov != None, "Must provide pre-estimated covariance matrix to estimate on chunks!"
	assert args.e != 0, "Must specify the number of eigenvectors to use in chunk-mode!"
	C = pd.read_csv(str(args.cov), sep="\t", header=None).as_matrix()
	nEV = args.e

	if args.genoInbreed:
		assert args.F != None, "Must provide pre-estimated inbreeding coefficients to call genotypes using inbreeding in chunks mode!"
		F = pd.read_csv(str(args.F), sep="\t", header=None).as_matrix()


# Standard PCAngsd approach
if args.chunksize == 0:

	# Parse Beagle file(s) for covariance matrix estimation 
	if args.beagle != None:
		print "Parsing Beagle file"
		likeDF = pd.read_csv(str(args.beagle), sep="\t")
		likeMatrix = likeDF.ix[:, 3:].as_matrix().T
		pos = likeDF.ix[:, 0]

	elif args.beaglelist != None:
		fBeagle = open(str(args.beaglelist), "r")
		likeList = fBeagle.readlines()
		print "Parsing " + str(len(likeList)) + " Beagle files"
		fBeagle.close()
		likeDF = pd.read_csv(likeList[0].replace("\n",""), sep="\t")
		print "Parsed 1/" + str(len(likeList))
		
		for i in range(1, len(likeList)):
			likeDF_2 = pd.read_csv(likeList[i].replace("\n",""), sep="\t")
			likeDF = pd.concat([likeDF, likeDF_2], ignore_index=True)
			print "Parsed " + str(i+1) + "/" + str(len(likeList))

		likeDF_2 = None
		likeMatrix = likeDF.ix[:, 3:].as_matrix().T
		pos = likeDF.ix[:, 0]


	# Estimate covariance matrix
	print "\n" + "Estimating population allele frequencies"
	f = alleleEM(likeMatrix, args.EM, args.EM_tole)
	mask = (f >= 0.05) & (f <= 0.95)
	print "Number of sites evaluated: " + str(np.sum(mask))

	# Update arrays
	f = f[mask]
	likeMatrix = likeMatrix[:, mask]

	# Reset position info dataframe to evaluated sites
	pos = pos.ix[mask]
	pos.reset_index(drop=True, inplace=True)

	# PCAngsd
	print "\n" + "Estimating covariance matrix"	
	C, indf, nEV, X, expG = PCAngsd(likeMatrix, args.e, args.M, f, args.M_tole, args.reg, args.scaled, args.LD)

	# Column names
	indNames = ["ind" + str(i) for i in range(indf.shape[0])]

	# Create data frames
	covDF = pd.DataFrame(C)
	fDF = pd.DataFrame(f, columns=["maf"])
	fDF.insert(0, "marker", pos)
	indfDF = pd.DataFrame(indf.T, columns=indNames)
	indfDF.insert(0, "marker", pos)

	# Save data frames
	covDF.to_csv(str(args.o) + ".cov", sep="\t", header=False, index=False)
	print "Saved covariance matrix as " + str(args.o) + ".cov"
	fDF.to_csv(str(args.o) + ".mafs.gz", sep="\t", index=False, compression="gzip")
	print "Saved population allele frequencies as " + str(args.o) + ".mafs.gz"
	indfDF.to_csv(str(args.o) + ".indmafs.gz", sep="\t", index=False, compression="gzip")
	print "Saved individual allele frequencies as " + str(args.o) + ".indmafs.gz"

	# Release memory
	covDF = None
	fDF = None
	indfDF = None

	if not param_selection:
		X = None
	elif param_selection and args.selection == 1:
		expG = None
	elif param_selection and args.selection == 2:
		X = expG


	# Selection scan
	if param_selection and args.selection == 1:
		print "\n" + "Performing selection scan using FastPCA method"

		# Perform selection scan and save data frame
		pcNames = ["PC" + str(pc) for pc in range(1, nEV+1)]
		chisqDF = pd.DataFrame(selectionScan(X, C, nEV, model=1).T, columns=pcNames)
		chisqDF.insert(0, "marker", pos)
		chisqDF.to_csv(str(args.o) + ".selection.gz", sep="\t", index=False, compression="gzip")
		print "Saved selection statistics for the top PCs as " + str(args.o) + ".selection.gz"

		# Release memory
		chisqDF = None

	elif param_selection and args.selection == 2:
		print "\n" + "Performing selection scan using PCAdapt method"

		# Perform selection scan and save data frame
		mahalanobisDF = pd.DataFrame(selectionScan(X, C, nEV, model=2), columns=["mahalanobis"])
		mahalanobisDF.insert(0, "marker", pos)
		mahalanobisDF.to_csv(str(args.o) + ".selection.gz", sep="\t", index=False, compression="gzip")
		print "Saved selection statistics for the top PCs as " + str(args.o) + ".selection.gz"

		# Release memory
		mahalanobisDF = None


	# Kinship estimation
	if param_kinship:
		print "\n" + "Estimating kinship matrix"

		# Perform kinship estimation
		phi = kinshipConomos(likeMatrix, indf)

		# Save data frame
		phiDF = pd.DataFrame(phi)
		phiDF.to_csv(str(args.o) + ".kinship", sep="\t", header=False, index=False)
		print "Saved kinship matrix as " + str(args.o) + ".kinship"

		# Release memory
		phiDF = None


	# Individual inbreeding coefficients
	if param_inbreed and args.inbreed == 1:
		print "\n" + "Estimating inbreeding coefficients using maximum likelihood estimator (EM)"

		# Estimating inbreeding coefficients
		F = inbreedEM(likeMatrix, indf, 1, args.inbreed_iter, args.inbreed_tole)

		# Save data frame
		F_DF = pd.DataFrame(F)
		F_DF.to_csv(str(args.o) + ".inbreed", sep="\t", header=False, index=False)
		print "Saved inbreeding coefficients as " + str(args.o) + ".inbreed"

		# Release memory
		if not args.genoInbreed:
			F = None
		F_DF = None

	elif param_inbreed and args.inbreed == 2:
		print "\n" + "Estimating inbreeding coefficients using Simple estimator (EM)"

		# Estimating inbreeding coefficients
		F = inbreedEM(likeMatrix, indf, 2, args.inbreed_iter, args.inbreed_tole)

		# Save data frame
		F_DF = pd.DataFrame(F)
		F_DF.to_csv(str(args.o) + ".inbreed", sep="\t", header=False, index=False)
		print "Saved inbreeding coefficients as " + str(args.o) + ".inbreed"

		# Release memory
		if not args.genoInbreed:
			F = None
		F_DF = None
		
	elif param_inbreed and args.inbreed == 3 and param_kinship:
		print "\n" + "Estimating inbreeding coefficients using kinship estimator (PC-Relate)"

		# Estimating inbreeding coefficients by previously estimated kinship matrix
		F = 2*phi.diagonal() - 1

		# Save data frame
		F_DF = pd.DataFrame(F)
		F_DF.to_csv(str(args.o) + ".inbreed", sep="\t", header=False, index=False)
		print "Saved inbreeding coefficients as " + str(args.o) + ".inbreed"

		# Release memory
		if not args.genoInbreed:
			F = None
		phi = None
		F_DF = None


	# Per-site inbreeding coefficients
	if param_inbreedSites:
		print "\n" + "Estimating per-site inbreeding coefficients using simple estimator (EM) and performing LRT"

		# Estimating per-site inbreeding coefficients
		Fsites, lrt = inbreedSitesEM(likeMatrix, indf, args.inbreed_iter, args.inbreed_tole)

		# Save data frames
		Fsites_DF = pd.DataFrame(Fsites, columns=["F"])
		Fsites_DF.insert(0, "marker", pos)
		Fsites_DF.to_csv(str(args.o) + ".inbreedSites.gz", sep="\t", index=False, compression="gzip")
		print "Saved per-site inbreeding coefficients as " + str(args.o) + ".inbreedSites.gz"

		lrt_DF = pd.DataFrame(lrt, columns=["chi2"])
		lrt_DF.insert(0, "marker", pos)
		lrt_DF.to_csv(str(args.o) + ".lrtSites.gz", sep="\t", index=False, compression="gzip")
		print "Saved likelihood ratio tests as " + str(args.o) + ".lrtSites.gz"

		# Release memory
		Fsites = None
		Fsites_DF = None
		lrt = None
		lrt_DF = None


	# Genotype calling
	if param_call:
		print "\n" + "Calling genotypes with a threshold of " + str(args.geno)

		# Call genotypes and save data frame
		genotypesDF = pd.DataFrame(callGeno(likeMatrix, indf, args.geno, None).T, columns=indNames)
		genotypesDF.insert(0, "marker", pos)
		genotypesDF.to_csv(str(args.o) + ".geno.gz", "\t", index=False, compression="gzip")
		print "Saved called genotypes as " + str(args.o) + ".geno.gz"

		# Release memory
		genotypesDF = None

	elif args.genoInbreed:
		print "\n" + "Calling genotypes with a threshold of " + str(args.genoInbreed)

		# Call genotypes and save data frame
		genotypesDF = pd.DataFrame(callGeno(likeMatrix, indf, args.genoInbreed, F).T, columns=indNames)
		genotypesDF.insert(0, "marker", pos)
		genotypesDF.to_csv(str(args.o) + ".genoInbreed.gz", "\t", index=False, compression="gzip")
		print "Saved called genotypes as " + str(args.o) + ".genoInbreed.gz"

		# Release memory
		genotypesDF = None


##### Reading in chunks of Beagle file
if args.chunksize > 0:
	print "Performing estimations in chunk-mode"
	likeReader = pd.read_csv(str(args.beagle), sep="\t", chunksize=args.chunksize)
	
	# Read in chunks of the genotype likelihoods
	c = 0
	for chunk in likeReader:
		likeMatrix = chunk.ix[:, 3:].as_matrix().T
		pos = chunk.ix[:, 0]

		# Estimate population allele frequencies
		print "\n" + "Estimating population allele frequencies"
		f = alleleEM(likeMatrix, args.EM, args.EM_tole)
		mask = (f >= 0.05) & (f <= 0.95)

		# Update arrays
		f = f[mask]
		likeMatrix = likeMatrix[:, mask]
		pos = pos.ix[mask]
		pos.reset_index(drop=True, inplace=True)

		# Estimate individual allele frequencies
		print "Estimating individual allele frequencies"
		indf, expG = individualF(likeMatrix, f, C, nEV, args.M, args.M_tole, args.reg, args.scaled)
		
		# Column names
		indNames = ["ind" + str(i) for i in range(indf.shape[0])]
		
		# Save data frames
		fDF = pd.DataFrame(f, columns=["maf"])
		fDF.insert(0, "marker", pos)
		indfDF = pd.DataFrame(indf.T, columns=indNames)
		indfDF.insert(0, "marker", pos)

		if c==0:
			fDF.to_csv(str(args.o) + ".mafs.gz", sep="\t", index=False, compression="gzip")
			print "Saved population allele frequencies as " + str(args.o) + ".mafs.gz"
			indfDF.to_csv(str(args.o) + ".indmafs.gz", sep="\t", index=False, compression="gzip")
			print "Saved individual allele frequencies as " + str(args.o) + ".indmafs.gz"
		else:
			fDF.to_csv(str(args.o) + ".mafs.gz", sep="\t", header=False, index=False, compression="gzip", mode="a")
			print "Saved population allele frequencies as " + str(args.o) + ".mafs.gz"
			indfDF.to_csv(str(args.o) + ".indmafs.gz", sep="\t", header=False, index=False, compression="gzip", mode="a")
			print "Saved individual allele frequencies as " + str(args.o) + ".indmafs.gz"

		# Selection scan
		if param_selection and args.selection == 1:
			print "Performing selection scan using FastPCA method"

			# Perform selection scan and save data frame
			X = (expG - 2*f)/np.sqrt(2*f*(1-f)) # Standardized genotype matrix
			pcNames = ["PC" + str(pc) for pc in range(1, nEV+1)]
			chisqDF = pd.DataFrame(selectionScan(X, C, nEV, model=1).T, columns=pcNames)
			chisqDF.insert(0, "marker", pos)

			if c == 0:
				chisqDF.to_csv(str(args.o) + ".selection.gz", sep="\t", index=False, compression="gzip")
			else:
				chisqDF.to_csv(str(args.o) + ".selection.gz", sep="\t", header=False, index=False, compression="gzip", mode="a")
			print "Saved selection statistics for the top PCs as " + str(args.o) + ".selection.gz"

			# Release memory
			chisqDF = None

		elif param_selection and args.selection == 2:
			print "Performing selection scan using PCAdapt method"

			# Perform selection scan and save data frame
			mahalanobisDF = pd.DataFrame(selectionScan(expG, C, nEV, model=2), columns=["mahalanobis"])
			mahalanobisDF.insert(0, "marker", pos)

			if c == 0:
				mahalanobisDF.to_csv(str(args.o) + ".selection.gz", sep="\t", index=False, compression="gzip")
			else:
				mahalanobisDF.to_csv(str(args.o) + ".selection.gz", sep="\t", header=False, index=False, compression="gzip", mode="a")
			print "Saved selection statistics for the top PCs as " + str(args.o) + ".selection.gz"

			# Release memory
			mahalanobisDF = None

		expG = None


		# Per-site inbreeding coefficients
		if param_inbreedSites:
			print "Estimating per-site inbreeding coefficients using simple estimator (EM) and performing LRT"

			# Estimating per-site inbreeding coefficients
			Fsites, lrt = inbreedSitesEM(likeMatrix, indf, args.inbreed_iter, args.inbreed_tole)

			# Save data frames
			Fsites_DF = pd.DataFrame(Fsites, columns=["F"])
			Fsites_DF.insert(0, "marker", pos)

			if c == 0:
				Fsites_DF.to_csv(str(args.o) + ".inbreedSites.gz", sep="\t", index=False, compression="gzip")
			else:
				Fsites_DF.to_csv(str(args.o) + ".inbreedSites.gz", sep="\t", header=False, index=False, compression="gzip", mode="a")
			print "Saved per-site inbreeding coefficients as " + str(args.o) + ".inbreedSites.gz"

			lrt_DF = pd.DataFrame(lrt, columns=["chi2"])
			lrt_DF.insert(0, "marker", pos)

			if c == 0:
				lrt_DF.to_csv(str(args.o) + ".lrtSites.gz", sep="\t", index=False, compression="gzip")
			else:
				lrt_DF.to_csv(str(args.o) + ".lrtSites.gz", sep="\t", header=False, index=False, compression="gzip", mode="a")
			print "Saved likelihood ratio tests as " + str(args.o) + ".lrtSites.gz"

			# Release memory
			Fsites = None
			Fsites_DF = None
			lrt = None
			lrt_DF = None

		# Genotype calling
		if param_call:
			print "Calling genotypes with a threshold of " + str(args.geno)

			# Call genotypes and save data frame
			genotypesDF = pd.DataFrame(callGeno(likeMatrix, indf, args.geno, None).T, columns=indNames)
			genotypesDF.insert(0, "marker", pos)

			if c == 0:
				genotypesDF.to_csv(str(args.o) + ".geno.gz", "\t", index=False, compression="gzip")
			else:
				genotypesDF.to_csv(str(args.o) + ".geno.gz", "\t", header=False, index=False, compression="gzip", mode="a")
			print "Saved called genotypes as " + str(args.o) + ".geno.gz"

			# Release memory
			genotypesDF = None

		elif args.genoInbreed:
			print "Calling genotypes with a threshold of " + str(args.genoInbreed)

			# Call genotypes and save data frame
			genotypesDF = pd.DataFrame(callGeno(likeMatrix, indf, args.genoInbreed, F).T, columns=indNames)
			genotypesDF.insert(0, "marker", pos)

			if c == 0:
				genotypesDF.to_csv(str(args.o) + ".genoInbreed.gz", "\t", index=False, compression="gzip")
			else:
				genotypesDF.to_csv(str(args.o) + ".genoInbreed.gz", "\t", header=False, index=False, compression="gzip", mode="a")
			print "Saved called genotypes as " + str(args.o) + ".genoInbreed.gz"

			# Release memory
			genotypesDF = None


		indf = None
		likeMatrix = None
		pos = None
		print "Processed chunk number " + str(c+1) + "\n"
		c += 1