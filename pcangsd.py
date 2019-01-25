"""
PCAngsd Framework: Population genetic analyses for NGS data using PCA. Main caller.
"""

__author__ = "Jonas Meisner"

# Import libraries
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import argparse
import numpy as np
import pandas as pd
import os

# Import functions
import emMAF
import covariance
import relateProject
import callGeno
import emInbreed
import emInbreedSites
import kinship
import selection
import admixture

##### Argparse #####
parser = argparse.ArgumentParser(prog="PCAngsd")
parser.add_argument("--version", action="version", version="%(prog)s 0.973")
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
	help="Call genotypes from posterior probabilities using individual allele frequencies and \
	inbreeding coefficients as prior")
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
parser.add_argument("-admix_K", metavar="INT", type=int,
	help="Number of ancestral population for admixture estimation")
parser.add_argument("-admix_iter", metavar="INT", type=int, default=100,
	help="Maximum iterations for admixture estimation - NMF (100)")
parser.add_argument("-admix_tole", metavar="FLOAT", type=float, default=1e-5,
	help="Tolerance for admixture estimation update - NMF (1e-5)")
parser.add_argument("-admix_batch", metavar="INT", type=int, default=5,
	help="Number of batches used for stochastic gradient descent (5)")
parser.add_argument("-admix_save", action="store_true",
	help="Save population-specific allele frequencies (Binary)")
parser.add_argument("-admix_auto", metavar="FLOAT", type=float,
	help="Automatic search for optimal alpha. Specify upper bound")
parser.add_argument("-admix_depth", metavar="INT", type=int, default=5,
	help="Depth of automatic search for alpha")
parser.add_argument("-remove_related", metavar="FILE",
	help="Remove related individuals based on input kinship matrix")
parser.add_argument("-related_tole", metavar="FLOAT", type=float, default=0.03125,
	help="Tolerance for kinship coefficient (0.03125)")
parser.add_argument("-relatedPCA", metavar="FILE",
	help="Input file for estimated kinship matrix")
parser.add_argument("-maf_save", action="store_true",
	help="Save estimated population allele frequencies (Binary)")
parser.add_argument("-indf_save", action="store_true",
	help="Save estimated individual allele frequencies (Binary)")
parser.add_argument("-expg_save", action="store_true",
	help="Save genotype dosages (Binary)")
parser.add_argument("-sites_save", action="store_true",
	help="Save marker IDs of filtered sites")
parser.add_argument("-post_save", action="store_true",
	help="Save posterior genotype probabilities in BEAGLE format")
parser.add_argument("-allocate_sites", metavar="INT", type=int, default=0,
	help="Pre-allocate memory for specified number of sites")
parser.add_argument("-threads", metavar="INT", type=int, default=1,
	help="Number of threads")
parser.add_argument("-o", metavar="OUTPUT", help="Prefix output file name", default="pcangsd")
args = parser.parse_args()

print "PCAngsd 0.973"
print "Using " + str(args.threads) + " thread(s)"

# Setting up workflow parameters
if (args.kinship) or (args.inbreed == 3):
	param_kinship = True
else:
	param_kinship = False

if args.genoInbreed is not None:
	assert param_inbreed, "Inbreeding coefficients must be estimated in order to use -genoInbreed! \
	Use -inbreed parameter!"

# Check parsing
if args.beagle is None:
	assert (args.plink is not None), "Missing input file! (-beagle or -plink)"
if (args.indf is not None) or (args.relatedPCA is not None):
	assert (args.e != 0), "Specify number of eigenvectors used to estimate allele frequencies! (-e)"
	e = args.e

# Individual filtering based on relatedness
if args.remove_related is not None:
	phi = np.genfromtxt(args.remove_related)

	# Boolean vector of unrelated individuals
	print "Masking related individuals with pair-wise kinship estimates >= " + str(args.related_tole)
	phi[np.triu_indices(phi.shape[0])] = 0 # Setting half of matrix to 0
	if len(np.unique(np.where(phi > args.related_tole)[0])) < len(np.unique(np.where(phi > args.related_tole)[1])):
		relatedIndices = np.unique(np.where(phi > args.related_tole)[0])
	else:
		relatedIndices = np.unique(np.where(phi > args.related_tole)[1])
	unrelatedI = np.isin(np.arange(phi.shape[0]), relatedIndices, invert=True)
	
	# Create or update index vector
	m = sum(unrelatedI)
	np.savetxt(args.o + ".keep", unrelatedI, fmt="%i")
	print "Keeping " + str(m) + " individuals after filtering (removing " + str(phi.shape[0] - m) + ")"
	print "Boolean vector of kept individuals saved as " + str(args.o) + ".keep" 
	unrelatedI = np.repeat(unrelatedI, 3)
	
	del phi, relatedIndices
else:
	unrelatedI = None


# Parse input file
if args.beagle is not None:
	print "\nParsing Beagle file"
	assert (os.path.isfile(args.beagle)), "Beagle file doesn't exist!"
	assert args.beagle[-3:] == ".gz", "Beagle file must be in gzip format!"
	from helpFunctions import readGzipBeagle

	if args.post_save or args.sites_save:
		pos_save = True
	else:
		pos_save = False
	likeMatrix, pos, alleleMatrix = readGzipBeagle(args.beagle, args.allocate_sites, unrelatedI, pos_save)
else:
	print "\nParsing PLINK files"
	from helpFunctions import readPlink
	likeMatrix, f, pos, indList = readPlink(args.plink, args.epsilon, args.threads)

m, n = likeMatrix.shape
m /= 3
print str(m) + " samples and " + str(n) + " sites"


##### Estimate population allele frequencies #####
if args.beagle is not None:
	print "\nEstimating population allele frequencies"
	f = emMAF.alleleEM(likeMatrix, args.maf_iter, args.maf_tole, args.threads)


##### Filtering sites
if args.minMaf > 0.0:
	maskMAF = (f >= args.minMaf) & (f <= 1-args.minMaf)
	print "Number of sites after MAF filtering (" + str(args.minMaf) + "): " + str(np.sum(maskMAF))
	
	# Update arrays
	f = np.compress(maskMAF, f)
	likeMatrix = np.compress(maskMAF, likeMatrix, axis=1)
	
	if not (args.post_save or args.sites_save):
		del maskMAF

if args.HWE_filter is not None:
	import scipy.stats as st
	lrtVec = pd.read_csv(args.HWE_filter, header=None, dtype=float, squeeze=True).as_matrix()
	maskHWE = st.chi2.sf(lrtVec, 1) > args.HWE_tole # Boolean vector for HWE statistics
	del lrtVec
	print "Number of sites after HWE filtering (" + str(args.HWE_tole) + "): " + str(np.sum(maskHWE))

	# Update arrays
	f = np.compress(maskHWE, f)
	likeMatrix = np.compress(maskHWE, likeMatrix, axis=1)

	if not (args.post_save or args.sites_save):
		del maskHWE


##### PCAngsd - Individual allele frequencies and covariance matrix #####
if args.relatedPCA is not None:
	print "\nPerforming PCAngsd taking relatedness into account using " + str(args.e) + " principal components"
	phi = np.genfromtxt(args.relatedPCA)
	C, Pi, E, V, f = relateProject.relatedPCAngsd(likeMatrix, args.e, f, phi, args.related_tole, args.iter, args.tole, args.threads)

	# Create and save data frames
	pd.DataFrame(C).to_csv(str(args.o) + ".unrelated.cov", sep="\t", header=False, index=False)
	print "Saved covariance matrix (unrelated) as " + str(args.o) + ".unrelated.cov"

	pd.DataFrame(V).to_csv(str(args.o) + ".eigenvecs", sep="\t", header=False, index=False)
	print "Saved combined eigenvectors as " + str(args.o) + ".eigenvecs"

elif args.indf is None:
	print "\nEstimating covariance matrix"
	C, Pi, e, E = covariance.PCAngsd(likeMatrix, args.e, args.iter, f, args.tole, args.threads)

	# Create and save data frames
	pd.DataFrame(C).to_csv(str(args.o) + ".cov", sep="\t", header=False, index=False)
	print "Saved covariance matrix as " + str(args.o) + ".cov"
else:
	print "\nParsing individual allele frequencies"
	Pi = np.load(args.indf).astype(np.float32, copy=False)


##### Selection scan #####
if args.selection is not None:
	assert args.relatedPCA is None, "Can't use projected samples in selection scan!"

	if args.indf is not None:
		print "Estimating genotype dosages and covariance matrix"
		E = np.empty(Pi.shape, dtype=np.float32)
		diagC = np.empty(m)

		# Multithreading parameters
		chunk_N = int(np.ceil(float(m)/args.threads))
		chunks = [i * chunk_N for i in xrange(args.threads)]

		# Multithreading
		threads = [threading.Thread(target=covariance.covPCAngsd, args=(likeMatrix, Pi, f, chunk, chunk_N, E, diagC)) for chunk in chunks]
		for thread in threads:
			thread.start()
		for thread in threads:
			thread.join()

		if args.altC is None:
			C = covariance.estimateCov(E, diagC, f, chunks, chunk_N)
		del diagC

	# Parse alternative covariance matrix if given
	if args.altC is not None:
		print "Parsing alternative covariance matrix"
		C = pd.read_csv(args.altC, sep="\t", header=None, dtype=np.float).as_matrix()
		assert (m == C.shape[0]), "Number of individuals must match for alternative covariance matrix!"

	if args.selection == 1:
		print "\nPerforming selection scan using FastPCA method"

		# Perform selection scan and save data frame
		sel = selection.selectionScan(E, f, C, e, model=1, t=args.threads).T
		pd.DataFrame(sel).to_csv(str(args.o) + ".selection.gz", sep="\t", header=False, index=False, compression="gzip")
		print "Saved selection statistics for the top PCs as " + str(args.o) + ".selection.gz"

		# Release memory
		del sel

	elif args.selection == 2:
		print "\nPerforming selection scan using PCAdapt method"

		# Perform selection scan and save data frame
		sel = selection.selectionScan(E, f, C, e, model=2, t=args.threads)
		pd.DataFrame(sel).to_csv(str(args.o) + ".selection.gz", sep="\t", header=False, index=False, compression="gzip")
		print "Saved selection statistics as " + str(args.o) + ".selection.gz"

		# Release memory
		del sel


##### Kinship estimation #####
if param_kinship:
	print "\nEstimating kinship matrix"

	# Perform kinship estimation
	phi = kinship.kinshipConomos(likeMatrix, Pi, E, args.threads)
	pd.DataFrame(phi).to_csv(str(args.o) + ".kinship", sep="\t", header=False, index=False)
	print "Saved kinship matrix as " + str(args.o) + ".kinship"


# Optional save of genotype dosages
if args.expg_save:
	np.save(str(args.o) + ".expg", E.astype(float, copy=False))
	print "Saved genotype dosages as " + str(args.o) + ".expg.npy (Binary)"

# Release memory
if (args.indf is None) or (args.selection is not None):
	del C, E


##### Individual inbreeding coefficients #####
if args.inbreed == 1:
	print "\nEstimating inbreeding coefficients using maximum likelihood estimator (EM)"

	# Estimating inbreeding coefficients
	if args.iter == 0:
		print "Using population allele frequencies (-iter 0), not taking structure into account"
		F = emInbreed.inbreedEM(likeMatrix, f, 1, args.inbreed_iter, args.inbreed_tole, args.threads)
		pd.DataFrame(F).to_csv(str(args.o) + ".inbreed", sep="\t", header=False, index=False)
		print "Saved inbreeding coefficients as " + str(args.o) + ".inbreed"
	else:
		F = emInbreed.inbreedEM(likeMatrix, Pi, 1, args.inbreed_iter, args.inbreed_tole, args.threads)
		pd.DataFrame(F).to_csv(str(args.o) + ".inbreed", sep="\t", header=False, index=False)
		print "Saved inbreeding coefficients as " + str(args.o) + ".inbreed"

	# Release memory
	del F

elif args.inbreed == 2:
	print "\nEstimating inbreeding coefficients using Simple estimator (EM)"

	# Estimating inbreeding coefficients
	if args.iter == 0:
		print "Using population allele frequencies (-iter 0), not taking structure into account"
		F = emInbreed.inbreedEM(likeMatrix, f, 2, args.inbreed_iter, args.inbreed_tole, args.threads)
		pd.DataFrame(F).to_csv(str(args.o) + ".inbreed", sep="\t", header=False, index=False)
		print "Saved inbreeding coefficients as " + str(args.o) + ".inbreed"
	else:
		F = emInbreed.inbreedEM(likeMatrix, Pi, 2, args.inbreed_iter, args.inbreed_tole, args.threads)
		pd.DataFrame(F).to_csv(str(args.o) + ".inbreed", sep="\t", header=False, index=False)
		print "Saved inbreeding coefficients as " + str(args.o) + ".inbreed"

	# Release memory
	del F

elif args.inbreed == 3:
	print "\nEstimating inbreeding coefficients using kinship estimator (PC-Relate)"

	# Estimating inbreeding coefficients by previously estimated kinship matrix
	F = 2*phi.diagonal() - 1
	pd.DataFrame(F).to_csv(str(args.o) + ".inbreed", sep="\t", header=False, index=False)
	print "Saved inbreeding coefficients as " + str(args.o) + ".inbreed"

	# Release memory
	del phi


##### Per-site inbreeding coefficients #####
if args.inbreedSites:
	print "\nEstimating per-site inbreeding coefficients using simple estimator (EM) and performing LRT"

	# Estimating per-site inbreeding coefficients
	Fsites, lrt = emInbreedSites.inbreedSitesEM(likeMatrix, Pi, args.inbreed_iter, args.inbreed_tole, args.threads)

	# Save data frames
	pd.DataFrame(Fsites).to_csv(str(args.o) + ".inbreed.sites.gz", header=False, index=False, compression="gzip")
	print "Saved per-site inbreeding coefficients as " + str(args.o) + ".inbreed.sites.gz"

	pd.DataFrame(lrt).to_csv(str(args.o) + ".lrt.sites.gz", sep="\t", header=False, index=False, compression="gzip")
	print "Saved likelihood ratio tests as " + str(args.o) + ".lrt.sites.gz"

	# Release memory
	del Fsites
	del lrt


##### Genotype calling #####
if args.geno is not None:
	print "\nCalling genotypes with a threshold of " + str(args.geno)

	# Call genotypes and save data frame
	G = callGeno.callGeno(likeMatrix, Pi, None, args.geno, args.threads).T
	np.save(str(args.o) + ".geno", G)
	print "Saved called genotypes as " + str(args.o) + ".geno.npy (Binary)"

	# Release memory
	del G

elif args.genoInbreed is not None:
	print "\nCalling genotypes with a threshold of " + str(args.genoInbreed)

	# Call genotypes and save data frame
	G = callGeno.callGeno(likeMatrix, Pi, F, args.genoInbreed, args.threads).T
	np.save(str(args.o) + ".genoInbreed", G)
	print "Saved called genotypes as " + str(args.o) + ".genoInbreed.npy (Binary)"

	# Release memory
	del G


##### Admixture proportions #####
if args.admix:
	if args.admix_K is not None:
		K = args.admix_K
	else:
		K = e + 1

	if args.admix_seed[0] is None:
		from time import time
		S_list = [int(time())]
	else:
		S_list = args.admix_seed

	# Automatic search for optimal alpha parameter
	if args.admix_auto is not None:
		print "\n" + ""
		Q_admix, F_admix, a_best = admixture.alphaSearch(args.admix_auto, args.admix_depth, Pi, K, likeMatrix, args.admix_iter, args.admix_tole, S_list[0], args.admix_batch, args.threads)
		pd.DataFrame(Q_admix).to_csv(str(args.o) + ".K" + str(K) + ".a" + str(a_best) + ".qopt", sep=" ", header=False, index=False)
		print "Saved admixture proportions as " + str(args.o) + ".K" + str(K) + ".a" + str(a_best) + ".qopt"

		if args.admix_save:
			if args.admix_seed[0] is None:
				np.save(str(args.o) + ".K" + str(K) + ".a" + str(a_best) + ".fopt", F_admix)
				print "Saved population-specific allele frequencies as " + str(args.o) + ".K" + str(K) + ".a" + str(a_best) + ".fopt.npy (Binary)"

	# Standard admixture estimation
	else:
		for a in args.admix_alpha:
			for s in S_list:
				print "\nEstimating admixture using NMF with K=" + str(K) + ", alpha=" + str(a) + ", batch=" + str(args.admix_batch) + " and seed=" + str(s)
				Q_admix, F_admix, _ = admixture.admixNMF(Pi, K, likeMatrix, a, args.admix_iter, args.admix_tole, s, args.admix_batch, args.threads)

				# Save data frame
				if args.admix_seed[0] is None:
					pd.DataFrame(Q_admix).to_csv(str(args.o) + ".K" + str(K) + ".a" + str(a) + ".qopt", sep=" ", header=False, index=False)
					print "Saved admixture proportions as " + str(args.o) + ".K" + str(K) + ".a" + str(a) + ".qopt"
				else:
					pd.DataFrame(Q_admix).to_csv(str(args.o) + ".K" + str(K) + ".a" + str(a) + ".s" + str(s) + ".qopt", sep=" ", header=False, index=False)
					print "Saved admixture proportions as " + str(args.o) + ".K" + str(K) + ".a" + str(a) + ".s" + str(s) + ".qopt"

				if args.admix_save:
					if args.admix_seed[0] is None:
						np.save(str(args.o) + ".K" + str(K) + ".a" + str(a) + ".fopt", F_admix)
						print "Saved population-specific allele frequencies as " + str(args.o) + ".K" + str(K) + ".a" + str(a) + ".fopt.npy (Binary)"
					else:
						np.save(str(args.o) + ".K" + str(K) + ".a" + str(a) + ".s" + str(s) + ".fopt", F_admix)
						print "Saved population-specific allele frequencies as " + str(args.o) + ".K" + str(K) + ".a" + str(a) + ".s" + str(s) + ".fopt.npy (Binary)"

				# Release memory
				del Q_admix
				del F_admix


##### Optional saves #####
# Save population allele frequencies
if args.maf_save:
	np.save(str(args.o) + ".maf", f)
	print "Saved population allele frequencies as " + str(args.o) + ".maf.npy (Binary)"

# Save individual allele frequencies
if args.indf_save:
	np.save(str(args.o) + ".indf", Pi.astype(float, copy=False))
	print "Saved individual allele frequencies as " + str(args.o) + ".indf.npy (Binary)"

# Updating sites info based on filters
if args.post_save or args.sites_save:
	if args.minMaf > 0.0:
		if args.HWE_filter is not None:
			pos = pos[maskMAF]
			pos = pos[maskHWE]
			if args.post_save:
				alleleMatrix = alleleMatrix[maskMAF, :]
				alleleMatrix = alleleMatrix[maskHWE, :]
			del maskMAF, maskHWE
		else:
			pos = pos[maskMAF]
			if args.post_save:
				alleleMatrix = alleleMatrix[maskMAF]
			del maskMAF
	elif args.HWE_filter is not None:
		pos = pos[maskHWE]
		if args.post_save:
			alleleMatrix = alleleMatrix[maskHWE]
		del maskHWE

# Save updated marker IDs
if args.sites_save:
	pd.DataFrame(pos).to_csv(str(args.o) + ".sites", header=False, index=False)
	print "Saved site IDs as " + str(args.o) + ".sites"

# Save posterior genotype probabilities
if args.post_save:
	print "\nSaving posterior genotype probabilities in Beagle format"
	from helpFunctions import writeReadBeagle
	
	if args.iter == 0:
		from helpFunctions import convertLikePostNoIndF
		print "Note: Using population allele frequencies as prior!"
		convertLikePostNoIndF(likeMatrix, f, args.threads)
	else:
		from helpFunctions import convertLikePost
		convertLikePost(likeMatrix, Pi, args.threads)
	
	if args.plink is None:
		writeReadBeagle(args.o + ".post.beagle", likeMatrix, pos, alleleMatrix)
	else:
		writeReadBeagle(args.o + ".post.beagle", likeMatrix, pos, None, indList)
	print "Saved posterior genotype probabilities as " + str(args.o) + ".post.beagle"