"""
PCAngsd Framework: Population genetic analyses for NGS data using PCA. Main caller.
"""

__author__ = "Jonas Meisner"

# Import functions
from emMAF import *
from covariance import *
from callGeno import *
from emInbreed import *
from emInbreedSites import *
from kinship import *
from selection import *
from admixture import *

# Import libraries
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import argparse
import numpy as np
import pandas as pd
import os.path

##### Argparse #####
parser = argparse.ArgumentParser(prog="PCAngsd")
parser.add_argument("--version", action="version", version="%(prog)s 0.92")
parser.add_argument("-beagle", metavar="FILE",
	help="Input file of genotype likelihoods in Beagle format (.gz)")
parser.add_argument("-indf", metavar="FILE",
	help="Input file of individual allele frequencies")
parser.add_argument("-plink", metavar="PLINK-PREFIX",
	help="Prefix for PLINK files (.bed, .bim, .fam)")
parser.add_argument("-epsilon", metavar="FLOAT", type=float, default=0.0,
	help="Assumption of error PLINK genotypes (0.0)")
parser.add_argument("-minMaf", metavar="FLOAT", type=float, default=0.05,
	help="Minimum minor allele frequency threshold (0.05)")
parser.add_argument("-iter", metavar="INT", type=int, default=100,
	help="Maximum iterations for estimation of individual allele frequencies (100)")
parser.add_argument("-tole", metavar="FLOAT", type=float, default=1e-5,
	help="Tolerance for update in estimation of individual allele frequencies (1e-5)")
parser.add_argument("-maf_iter", metavar="INT", type=int, default=200,
	help="Maximum iterations for population allele frequencies estimation - EM (200)")
parser.add_argument("-maf_tole", metavar="FLOAT", type=float, default=1e-4,
	help="Tolerance for population allele frequencies estimation update - EM (1e-4)")
parser.add_argument("-e", metavar="INT", type=int, default=0,
	help="Manual selection of eigenvectors used for SVD")
parser.add_argument("-geno", metavar="FLOAT", type=float,
	help="Call genotypes from posterior probabilities using individual allele frequencies as prior")
parser.add_argument("-genoInbreed", metavar="FLOAT", type=float,
	help="Call genotypes from posterior probabilities using individual allele frequencies and inbreeding coefficients as prior")
parser.add_argument("-inbreed", metavar="INT", type=int,
	help="Compute the per-individual inbreeding coefficients by specified model")
parser.add_argument("-inbreedSites", action="store_true",
	help="Compute the per-site inbreeding coefficients by specified model and LRT")
parser.add_argument("-inbreed_iter", metavar="INT", type=int, default=200,
	help="Maximum iterations for inbreeding coefficients estimation - EM (200)")
parser.add_argument("-inbreed_tole", metavar="FLOAT", type=float, default=1e-4,
	help="Tolerance for inbreeding coefficients estimation update - EM (1e-4)")
parser.add_argument("-HWE_filter", metavar="FILE",
	help="Input file of LRT values in sites from HWE test")
parser.add_argument("-HWE_tole", metavar="FLOAT", type=float, default=1e-6,
	help="Threshold in HWE test (1e-6)")
parser.add_argument("-selection", metavar="INT", type=int,
	help="Perform selection scan using the top principal components by specified model")
parser.add_argument("-altC", metavar="FILE",
	help="Input file for alternative covariance matrix to use in selection scan")
parser.add_argument("-kinship", action="store_true",
	help="Estimate the kinship matrix")
parser.add_argument("-admix", action="store_true",
	help="Estimate admixture proportions using NMF")
parser.add_argument("-admix_alpha", metavar="FLOAT-LIST", type=float, nargs="+", default=[0],
	help="Sparseness parameter for NMF in estimation of admixture proportions")
parser.add_argument("-admix_seed", metavar="INT-LIST", type=int, nargs="+", default=[None],
	help="Random seed for admixture estimation")
parser.add_argument("-admix_K", metavar="INT-LIST", type=int, nargs="+", default=[0],
	help="Number of ancestral population for admixture estimation")
parser.add_argument("-admix_iter", metavar="INT", type=int, default=50,
	help="Maximum iterations for admixture estimation - NMF (50)")
parser.add_argument("-admix_tole", metavar="FLOAT", type=float, default=1e-4,
	help="Tolerance for admixture estimation update - NMF (1e-4)")
parser.add_argument("-admix_batch", metavar="INT", type=int, default=5,
	help="Number of batches used for stochastic gradient descent (5)")
parser.add_argument("-admix_save", action="store_true",
	help="Save population-specific allele frequencies (Binary)")
parser.add_argument("-indf_save", action="store_true",
	help="Save estimated allele frequencies (Binary)")
parser.add_argument("-expg_save", action="store_true",
	help="Save genotype dosages (Binary)")
parser.add_argument("-sites_save", action="store_true",
	help="Save marker IDs of filtered sites")
parser.add_argument("-threads", metavar="INT", type=int, default=1,
	help="Number of threads")
parser.add_argument("-o", metavar="OUTPUT", help="Prefix output file name", default="pcangsd")
args = parser.parse_args()

print "PCAngsd 0.92"
print "Using " + str(args.threads) + " thread(s)"

# Setting up workflow parameters
if args.inbreed == 3:
	param_kinship = True

if args.kinship:
	param_kinship = True
else:
	param_kinship = False

if args.genoInbreed != None:
	assert param_inbreed, "Inbreeding coefficients must be estimated in order to use -genoInbreed! Use -inbreed parameter!"

# Check parsing
if args.beagle == None:
	assert (args.plink != None), "Missing input file! (-beagle or -plink)"
if args.indf != None:
	assert (args.e != 0), "Specify number of eigenvectors used to estimate allele frequencies! (-e)"

# Parse Beagle file
if args.beagle != None:
	print "\n" + "Parsing Beagle file"
	assert (os.path.isfile(args.beagle)), "Beagle file doesn't exist!"
	assert args.beagle[-3:] == ".gz", "Beagle file must be in gzip format!"
	from helpFunctions import readGzipBeagle
	likeMatrix = readGzipBeagle(args.beagle)
else:
	print "\n" + "Parsing PLINK files"
	from helpFunctions import readPlink
	(likeMatrix, f, pos) = readPlink(args.plink, args.epsilon, args.threads)

m, n = likeMatrix.shape
m /= 3
print str(m) + " samples and " + str(n) + " sites"


##### Estimate population allele frequencies #####
if args.beagle != None:
	print "Estimating population allele frequencies"
	f = alleleEM(likeMatrix, args.maf_iter, args.maf_tole, args.threads)


##### Filtering sites
if args.minMaf > 0.0:
	mask = (f >= args.minMaf) & (f <= 1-args.minMaf)
	print "Number of sites after MAF filtering (" + str(args.minMaf) + "): " + str(np.sum(mask))

	# Update arrays
	f = np.compress(mask, f)
	likeMatrix = np.compress(mask, likeMatrix, axis=1)

if args.HWE_filter != None:
	import scipy.stats as st
	lrtVec = pd.read_csv(args.HWE_filter, header=None, dtype=np.float, squeeze=True).as_matrix()
	boolVec = st.chi2.sf(lrtVec, 1) > args.HWE_tole
	del lrtVec
	print "Number of sites after HWE filtering (" + str(args.HWE_tole) + "): " + str(np.sum(boolVec))

	# Update arrays
	f = np.compress(boolVec, f)
	likeMatrix = np.compress(boolVec, likeMatrix, axis=1)


##### PCAngsd - Individual allele frequencies and covariance matrix #####
if args.indf == None:
	print "\n" + "Estimating covariance matrix"
	C, indf, nEV, expG = PCAngsd(likeMatrix, args.e, args.iter, f, args.tole, args.threads)

	# Create and save data frames
	pd.DataFrame(C).to_csv(str(args.o) + ".cov", sep="\t", header=False, index=False)
	print "Saved covariance matrix as " + str(args.o) + ".cov"
else:
	print "\n" + "Parsing individual allele frequencies"
	indf = np.load(args.indf)
	nEV = args.e


##### Selection scan #####
if args.selection != None:
	if args.indf != None:
		print "Estimating genotype dosages and covariance matrix"
		chunk_N = int(np.ceil(float(m)/args.threads))
		chunks = [i * chunk_N for i in xrange(args.threads)]
		expG = np.empty(indf.shape, dtype=np.float32)
		diagC = np.empty(m)

		# Multithreading
		threads = [threading.Thread(target=covPCAngsd, args=(likeMatrix, indf, f, chunk, chunk_N, expG, diagC)) for chunk in chunks]
		for thread in threads:
			thread.start()
		for thread in threads:
			thread.join()

		if args.altC == None:
			C = estimateCov(expG, diagC, f, chunks, chunk_N)
		del diagC

	# Parse alternative covariance matrix if given
	if args.altC != None:
		print "Parsing alternative covariance matrix"
		C = pd.read_csv(args.altC, sep="\t", header=None, dtype=np.float).as_matrix()
		assert (m == C.shape[0]), "Number of individuals must match for alternative covariance matrix!"


	if args.selection == 1:
		print "\n" + "Performing selection scan using FastPCA method"

		# Perform selection scan and save data frame
		chisqDF = pd.DataFrame(selectionScan(expG, f, C, nEV, model=1, threads=args.threads).T)
		chisqDF.to_csv(str(args.o) + ".selection.gz", sep="\t", header=False, index=False, compression="gzip")
		print "Saved selection statistics for the top PCs as " + str(args.o) + ".selection.gz"

		# Release memory
		del chisqDF

	elif args.selection == 2:
		print "\n" + "Performing selection scan using PCAdapt method"

		# Perform selection scan and save data frame
		mahalanobisDF = pd.DataFrame(selectionScan(expG, f, C, nEV, model=2, threads=args.threads))
		mahalanobisDF.to_csv(str(args.o) + ".selection.gz", sep="\t", header=False, index=False, compression="gzip")
		print "Saved selection statistics as " + str(args.o) + ".selection.gz"

		# Release memory
		del mahalanobisDF


##### Kinship estimation #####
if param_kinship:
	print "\n" + "Estimating kinship matrix"

	# Perform kinship estimation
	phi = kinshipConomos(likeMatrix, indf, expG, args.threads)
	pd.DataFrame(phi).to_csv(str(args.o) + ".kinship", sep="\t", header=False, index=False)
	print "Saved kinship matrix as " + str(args.o) + ".kinship"


# Optional save of genotype dosages
if args.expg_save:
	np.save(str(args.o) + ".expg", expG)
	print "Saved genotype dosages as " + str(args.o) + ".expg.npy (Binary)"

# Release memory
if (args.indf == None) or (args.selection != None):
	del C, expG


##### Individual inbreeding coefficients #####
if args.inbreed == 1:
	print "\n" + "Estimating inbreeding coefficients using maximum likelihood estimator (EM)"

	# Estimating inbreeding coefficients
	if args.iter == 0:
		print "Using population allele frequencies (-iter 0), not taking structure into account"
		F = inbreedEM(likeMatrix, f, 1, args.inbreed_iter, args.inbreed_tole)
		pd.DataFrame(F).to_csv(str(args.o) + ".inbreed", sep="\t", header=False, index=False)
		print "Saved inbreeding coefficients as " + str(args.o) + ".inbreed"
	else:
		F = inbreedEM(likeMatrix, indf, 1, args.inbreed_iter, args.inbreed_tole)
		pd.DataFrame(F).to_csv(str(args.o) + ".inbreed", sep="\t", header=False, index=False)
		print "Saved inbreeding coefficients as " + str(args.o) + ".inbreed"

	# Release memory
	del F

elif args.inbreed == 2:
	print "\n" + "Estimating inbreeding coefficients using Simple estimator (EM)"

	# Estimating inbreeding coefficients
	if args.iter == 0:
		print "Using population allele frequencies (-iter 0), not taking structure into account"
		F = inbreedEM(likeMatrix, f, 2, args.inbreed_iter, args.inbreed_tole)
		pd.DataFrame(F).to_csv(str(args.o) + ".inbreed", sep="\t", header=False, index=False)
		print "Saved inbreeding coefficients as " + str(args.o) + ".inbreed"
	else:
		F = inbreedEM(likeMatrix, indf, 2, args.inbreed_iter, args.inbreed_tole)
		pd.DataFrame(F).to_csv(str(args.o) + ".inbreed", sep="\t", header=False, index=False)
		print "Saved inbreeding coefficients as " + str(args.o) + ".inbreed"

	# Release memory
	del F

elif args.inbreed == 3:
	print "\n" + "Estimating inbreeding coefficients using kinship estimator (PC-Relate)"

	# Estimating inbreeding coefficients by previously estimated kinship matrix
	F = 2*phi.diagonal() - 1
	pd.DataFrame(F).to_csv(str(args.o) + ".inbreed", sep="\t", header=False, index=False)
	print "Saved inbreeding coefficients as " + str(args.o) + ".inbreed"

	# Release memory
	del phi

del f


##### Per-site inbreeding coefficients #####
if args.inbreedSites:
	print "\n" + "Estimating per-site inbreeding coefficients using simple estimator (EM) and performing LRT"

	# Estimating per-site inbreeding coefficients
	Fsites, lrt = inbreedSitesEM(likeMatrix, indf, args.inbreed_iter, args.inbreed_tole)

	# Save data frames
	pd.DataFrame(Fsites).to_csv(str(args.o) + ".inbreed.sites.gz", header=False, index=False, compression="gzip")
	print "Saved per-site inbreeding coefficients as " + str(args.o) + ".inbreed.sites.gz"

	pd.DataFrame(lrt).to_csv(str(args.o) + ".lrt.sites.gz", sep="\t", header=False, index=False, compression="gzip")
	print "Saved likelihood ratio tests as " + str(args.o) + ".lrt.sites.gz"

	# Release memory
	del Fsites
	del lrt


##### Genotype calling #####
if args.geno != None:
	print "\n" + "Calling genotypes with a threshold of " + str(args.geno)

	# Call genotypes and save data frame
	genotypesDF = pd.DataFrame(callGeno(likeMatrix, indf, None, args.geno, args.threads).T)
	genotypesDF.to_csv(str(args.o) + ".geno.gz", "\t", header=False, index=False, compression="gzip")
	print "Saved called genotypes as " + str(args.o) + ".geno.gz"

	# Release memory
	del genotypesDF

elif args.genoInbreed != None:
	print "\n" + "Calling genotypes with a threshold of " + str(args.genoInbreed)

	# Call genotypes and save data frame
	genotypesDF = pd.DataFrame(callGeno(likeMatrix, indf, F, args.genoInbreed, args.threads).T)
	genotypesDF.to_csv(str(args.o) + ".genoInbreed.gz", "\t", header=False, index=False, compression="gzip")
	print "Saved called genotypes as " + str(args.o) + ".genoInbreed.gz"

	# Release memory
	del genotypesDF


##### Admixture proportions #####
if args.admix:
	if args.admix_K[0] == 0:
		K_list = [nEV + 1]
	else:
		K_list = args.admix_K

	if args.admix_seed[0] == None:
		from time import time
		S_list = [int(time())]
	else:
		S_list = args.admix_seed

	for K in K_list:
		for a in args.admix_alpha:
			for s in S_list:
				print "\n" + "Estimating admixture using NMF with K=" + str(K) + ", alpha=" + str(a) + ", batch=" + str(args.admix_batch) + " and seed=" + str(s)
				Q_admix, F_admix = admixNMF(indf, K, likeMatrix, a, args.admix_iter, args.admix_tole, s, args.admix_batch, args.threads)

				# Save data frame
				if args.admix_seed[0] == None:
					pd.DataFrame(Q_admix).to_csv(str(args.o) + ".K" + str(K) + ".a" + str(a) + ".qopt", sep=" ", header=False, index=False)
					print "Saved admixture proportions as " + str(args.o) + ".K" + str(K) + ".a" + str(a) + ".qopt"
				else:
					pd.DataFrame(Q_admix).to_csv(str(args.o) + ".K" + str(K) + ".a" + str(a) + ".s" + str(s) + ".qopt", sep=" ", header=False, index=False)
					print "Saved admixture proportions as " + str(args.o) + ".K" + str(K) + ".a" + str(a) + ".s" + str(s) + ".qopt"

				if args.admix_save:
					if args.admix_seed[0] == None:
						np.save(str(args.o) + ".K" + str(K) + ".a" + str(a) + ".fopt", F_admix)
						print "Saved population-specific allele frequencies as " + str(args.o) + ".K" + str(K) + ".a" + str(a) + ".fopt.npy (Binary)"
					else:
						np.save(str(args.o) + ".K" + str(K) + ".a" + str(a) + ".s" + str(s) + ".fopt", F_admix)
						print "Saved population-specific allele frequencies as " + str(args.o) + ".K" + str(K) + ".a" + str(a) + ".s" + str(s) + ".fopt.npy (Binary)"

				# Release memory
				del Q_admix
				del F_admix


##### Optional saves #####
# Save individual allele frequencies
if args.indf_save:
	np.save(str(args.o) + ".indf", indf)
	print "Saved individual allele frequencies as " + str(args.o) + ".indf.npy (Binary)"

# Save updated marker IDs
if args.sites_save:
	if args.plink == None:
		pos = pd.read_csv(str(args.beagle), sep="\t", engine="c", header=None, skiprows=1, usecols=[0], compression="gzip")
		if args.minMaf > 0.0:
			pos = pos.ix[mask]
			del mask
		if args.HWE_filter != None:
			pos = pos.ix[boolVec]
			del boolVec
		pos.to_csv(str(args.o) + ".sites", header=False, index=False)
	else:
		if args.minMaf > 0.0:
			pos = pos[mask]
			del mask
		if args.HWE_filter != None:
			pos = pos[boolVec]
			del boolVec
		pd.DataFrame(pos).to_csv(str(args.o) + ".sites", header=False, index=False)

	print "Saved site IDs as " + str(args.o) + ".sites"