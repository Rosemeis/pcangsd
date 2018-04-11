"""
PCAngsd Framework: Population genetic analyses for NGS data using PCA. Main caller.
"""

__author__ = "Jonas Meisner"

# Import functions
from helpFunctions import *
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

##### Argparse #####
parser = argparse.ArgumentParser(prog="PCAngsd")
parser.add_argument("--version", action="version", version="%(prog)s 0.9")
parser.add_argument("-beagle", metavar="FILE", 
	help="Input file of genotype likelihoods in Beagle format")
parser.add_argument("-indf", metavar="FILE",
	help="Input file of individual allele frequencies")
parser.add_argument("-plink", metavar="PLINK-PREFIX",
	help="Prefix for PLINK files (.bed, .bim, .fam)")
parser.add_argument("-n", metavar="INT", type=int,
	help="Number of individuals")
parser.add_argument("-epsilon", metavar="FLOAT", type=float, default=0.0,
	help="Assumption of error PLINK genotypes (0.0)")
parser.add_argument("-minMaf", metavar="FLOAT", type=float, default=0.05,
	help="Minimum minor allele frequency threshold (0.05)")
parser.add_argument("-iter", metavar="INT", type=int, default=100,
	help="Maximum iterations for estimation of individual allele frequencies (100)")
parser.add_argument("-tole", metavar="FLOAT", type=float, default=5e-5,
	help="Tolerance for update in estimation of individual allele frequencies (5e-5)")
parser.add_argument("-maf_iter", metavar="INT", type=int, default=200,
	help="Maximum iterations for population allele frequencies estimation - EM (200)")
parser.add_argument("-maf_tole", metavar="FLOAT", type=float, default=5e-5,
	help="Tolerance for population allele frequencies estimation update - EM (5e-5)")
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
parser.add_argument("-inbreed_tole", metavar="FLOAT", type=float, default=5e-5,
	help="Tolerance for inbreeding coefficients estimation update - EM (5e-5)")
parser.add_argument("-selection", metavar="INT", type=int,
	help="Perform selection scan using the top principal components by specified model")
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
parser.add_argument("-admix_tole", metavar="FLOAT", type=float, default=5e-5,
	help="Tolerance for admixture estimation update - EM (5e-5)")
parser.add_argument("-admix_batch", metavar="INT", type=int, default=5,
	help="Number of batches used for stochastic gradient descent (5)")
parser.add_argument("-admix_save", action="store_true",
	help="Save population-specific allele frequencies (Binary)")
parser.add_argument("-freq_save", action="store_true",
	help="Save estimated allele frequencies (Binary)")
parser.add_argument("-sites_save", action="store_true",
	help="Save marker IDs of filtered sites")
parser.add_argument("-threads", metavar="INT", type=int, default=1,
	help="Number of threads")
parser.add_argument("-o", metavar="OUTPUT", help="Prefix output file name", default="pcangsd")
args = parser.parse_args()

print "Running PCAngsd with " + str(args.threads) + " thread(s)"

# Setting up workflow parameters
param_inbreed = False
param_selection = False
param_kinship = False

if args.selection != None:
	param_selection = True

if args.inbreed != None:
	param_inbreed = True
	if args.inbreed == 3:
		param_kinship = True

if args.kinship:
	param_kinship = True

if args.genoInbreed != None:
	assert param_inbreed, "Inbreeding coefficients must be estimated in order to use -genoInbreed! Use -inbreed parameter!"

# Check parsing
if args.plink == None:
	assert (args.beagle != None), "Missing input file! (-beagle or -plink)"
assert (args.n != None), "Specify number of individuals! (-n)"
if (args.indf != None):
	assert (args.e != 0), "Specify number of eigenvectors used to estimate allele frequencies!"

# Parse Beagle file
if args.plink == None:
	print "Parsing Beagle file"
	likeMatrix = pd.read_csv(str(args.beagle), sep="\t", engine="c", header=0, usecols=range(3, 3 + 3*args.n), dtype=np.float32, compression="gzip")
	likeMatrix = likeMatrix.as_matrix().T
else:
	chunk_N = int(np.ceil(float(args.n)/args.threads))
	chunks = [i * chunk_N for i in xrange(args.threads)]
	print "Parsing PLINK files"
	from pysnptools.snpreader import Bed # Import Microsoft Genomics PLINK reader
	snpClass = Bed(args.plink, count_A1=True)
	pos = np.copy(snpClass.sid)
	snpFile = snpClass.read(dtype=np.float32) # Read PLINK files into memory
	f = np.nanmean(snpFile.val, axis=0, dtype=np.float64)/2
	likeMatrix = np.zeros((3*args.n, snpFile.val.shape[1]), dtype=np.float32)
	print "Converting PLINK files into genotype likelihood matrix"

	# Multithreading
	threads = [threading.Thread(target=convertPlink, args=(likeMatrix, snpFile.val, chunk, chunk_N, args.epsilon)) for chunk in chunks]
	for thread in threads:
		thread.start()
	for thread in threads:
		thread.join()

	del snpClass, snpFile

##### Estimate population allele frequencies #####
if args.plink == None:
	print "\n" + "Estimating population allele frequencies"
	f = alleleEM(likeMatrix, args.maf_iter, args.maf_tole, args.threads)

if args.minMaf > 0.0:
	mask = (f >= args.minMaf) & (f <= 1-args.minMaf)
	print "Number of sites after filtering: " + str(np.sum(mask))

	# Update arrays
	f = np.compress(mask, f)
	likeMatrix = np.compress(mask, likeMatrix, axis=1)


##### PCAngsd - Individual allele frequencies and covariance matrix #####
if args.indf == None:
	print "\n" + "Estimating covariance matrix"	
	C, indf, nEV, expG = PCAngsd(likeMatrix, args.e, args.iter, f, args.tole, args.threads)

	# Create and save data frames
	pd.DataFrame(C).to_csv(str(args.o) + ".cov", sep="\t", header=False, index=False)
	print "Saved covariance matrix as " + str(args.o) + ".cov"
	if not param_selection:
		del C, expG

else:
	print "\n" + "Parsing individual allele frequencies"
	indf = np.fromfile(args.indf, dtype=np.float32, sep="").reshape(args.n, likeMatrix.shape[1])
	nEV = args.e


##### Selection scan #####
if param_selection:
	if args.indf != None:
		print "Estimating genotype dosages and covariance matrix"
		chunk_N = int(np.ceil(float(args.n)/args.threads))
		chunks = [i * chunk_N for i in xrange(args.threads)]
		expG = np.zeros(indf.shape, dtype=np.float32)
		diagC = np.zeros(args.n)

		# Multithreading
		threads = [threading.Thread(target=covPCAngsd, args=(likeMatrix, indf, f, chunk, chunk_N, expG, diagC)) for chunk in chunks]
		for thread in threads:
			thread.start()
		for thread in threads:
			thread.join()

		C = estimateCov(expG, diagC, f, chunks, chunk_N)

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
		print "Saved selection statistics for the top PCs as " + str(args.o) + ".selection.gz"

		# Release memory
		del mahalanobisDF

	del C, expG


##### Kinship estimation #####
if param_kinship:
	print "\n" + "Estimating kinship matrix"

	# Perform kinship estimation
	phi = kinshipConomos(likeMatrix, indf)
	pd.DataFrame(phi).to_csv(str(args.o) + ".kinship", sep="\t", header=False, index=False)
	print "Saved kinship matrix as " + str(args.o) + ".kinship"


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
	Fsites_DF = pd.DataFrame(Fsites)
	Fsites_DF.to_csv(str(args.o) + ".inbreedSites.gz", sep="\t", header=False, index=False, compression="gzip")
	print "Saved per-site inbreeding coefficients as " + str(args.o) + ".inbreedSites.gz"

	lrt_DF = pd.DataFrame(lrt)
	lrt_DF.to_csv(str(args.o) + ".lrtSites.gz", sep="\t", header=False, index=False, compression="gzip")
	print "Saved likelihood ratio tests as " + str(args.o) + ".lrtSites.gz"

	# Release memory
	del Fsites
	del Fsites_DF
	del lrt
	del lrt_DF


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
						F_admix.tofile(str(args.o) + ".K" + str(K) + ".a" + str(a) + ".fopt", sep="")
						print "Saved population-specific allele frequencies as " + str(args.o) + ".K" + str(K) + ".a" + str(a) + ".fopt (Binary)"
					else:
						F_admix.tofile(str(args.o) + ".K" + str(K) + ".a" + str(a) + ".s" + str(s) + ".fopt", sep="")
						print "Saved population-specific allele frequencies as " + str(args.o) + ".K" + str(K) + ".a" + str(a) + ".s" + str(s) + ".fopt (Binary)"

				# Release memory
				del Q_admix
				del F_admix


##### Optional saves #####
# Save updated marker IDs
if args.sites_save:
	if args.plink == None:
		pos = pd.read_csv(str(args.beagle), sep="\t", engine="c", header=0, usecols=[0], compression="gzip")
		if args.minMaf > 0.0:
			pos = pos.ix[mask]
			del mask
		pos.to_csv(str(args.o) + ".sites", header=False, index=False)
	else:
		if args.minMaf > 0.0:
			pd.DataFrame(pos[mask]).to_csv(str(args.o) + ".sites", header=False, index=False)
			del mask
		else:
			pd.DataFrame(pos).to_csv(str(args.o) + ".sites", header=False, index=False)

	print "Saved site IDs as " + str(args.o) + ".sites"
	del pos

# Save frequencies arrays
if args.freq_save:
	indf.tofile(str(args.o) + ".indf", sep="")
	print "Saved individual allele frequencies as " + str(args.o) + ".indf (Binary)"