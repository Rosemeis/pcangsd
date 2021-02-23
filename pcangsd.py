"""
PCAngsd.
Main caller.
"""

__author__ = "Jonas Meisner"

# Argparse
import argparse
parser = argparse.ArgumentParser(prog="PCAngsd")
parser.add_argument("--version", action="version", version="%(prog)s 1.01")
parser.add_argument("-beagle", metavar="FILE",
	help="Filepath to genotype likelihoods in gzipped Beagle format from ANGSD")
parser.add_argument("-filter", metavar="FILE",
	help="Input file of vector for filtering individuals")
parser.add_argument("-plink", metavar="FILE-PREFIX",
	help="Prefix PLINK files (.bed, .bim, .fam)")
parser.add_argument("-plink_error", metavar="FLOAT", type=float, default=0.0,
	help="Incorporate errors into genotypes")
parser.add_argument("-minMaf", metavar="FLOAT", type=float, default=0.05,
	help="Minimum minor allele frequency threshold (0.05)")
parser.add_argument("-maf_iter", metavar="INT", type=int, default=200,
	help="Maximum iterations for minor allele frequencies estimation - EM (200)")
parser.add_argument("-maf_tole", metavar="FLOAT", type=float, default=1e-4,
	help="Tolerance for minor allele frequencies estimation update - EM (1e-4)")
parser.add_argument("-iter", metavar="INT", type=int, default=100,
	help="Maximum iterations for estimation of individual allele frequencies (100)")
parser.add_argument("-tole", metavar="FLOAT", type=float, default=1e-5,
	help="Tolerance for update in estimation of individual allele frequencies (1e-5)")
parser.add_argument("-hwe", metavar="FILE",
	help="Input file of LRTs from HWE test for site filtering")
parser.add_argument("-hwe_tole", metavar="FLOAT", type=float, default=1e-6,
	help="Tolerance for HWE filtering of sites")
parser.add_argument("-e", metavar="INT", type=int, default=0,
	help="Manual selection of eigenvectors in modelling")
parser.add_argument("-pi", metavar="FILE",
	help="Load previous estimation of individual allele frequencies (.pi.npy)")
parser.add_argument("-selection", action="store_true",
	help="Perform PC-based genome-wide selection scan")
parser.add_argument("-snp_weights", action="store_true",
	help="Estimate SNP weights")
parser.add_argument("-pcadapt", action="store_true",
	help="Perform pcadapt selection scan")
parser.add_argument("-selection_e", metavar="INT", type=int, default=0,
	help="Manual selection of eigenvectors in selection scans/SNP weights")
parser.add_argument("-inbreedSites", action="store_true",
	help="Compute the per-site inbreeding coefficients and LRT")
parser.add_argument("-inbreedSamples", action="store_true",
	help="Compute the per-individual inbreeding coefficients")
parser.add_argument("-inbreed_iter", metavar="INT", type=int, default=200,
	help="Maximum iterations for inbreeding coefficients estimation - EM (200)")
parser.add_argument("-inbreed_tole", metavar="FLOAT", type=float, default=1e-4,
	help="Tolerance for inbreeding coefficients estimation update - EM (1e-4)")
parser.add_argument("-geno", metavar="FLOAT", type=float,
	help="Call genotypes from posterior probabilities with threshold")
parser.add_argument("-genoInbreed", metavar="FLOAT", type=float,
	help="Call genotypes (inbreeding) from posterior probabilities with threshold")
parser.add_argument("-admix", action="store_true",
	help="Estimate admixture proportions and ancestral allele frequencies")
parser.add_argument("-admix_K", metavar="INT", type=int,
	help="Number of admixture components")
parser.add_argument("-admix_iter", metavar="INT", type=int, default=200,
	help="Maximum number of iterations for admixture estimations - NMF")
parser.add_argument("-admix_tole", metavar="FLOAT", type=float, default=1e-5,
	help="Tolerance for admixture estimations - NMF (1e-5)")
parser.add_argument("-admix_batch", metavar="INT", type=int, default=10,
	help="Number of mini-batches in stochastic admixture estimations - NMF (10)")
parser.add_argument("-admix_alpha", metavar="FLOAT", type=float, default=0,
	help="Alpha value for regularization in admixture estimations - NMF (0)")
parser.add_argument("-admix_seed", metavar="INT", type=int, default=0,
	help="Random sede used in admixture estimations - NMF (0)")
parser.add_argument("-admix_auto", metavar="FLOAT", type=float,
	help="Enable automatic search for alpha by giving soft upper search bound")
parser.add_argument("-admix_depth", metavar="INT", type=int, default=7,
	help="Depth in automatic search of alpha")
parser.add_argument("-tree", action="store_true",
	help="Construct NJ tree from covariance matrix")
parser.add_argument("-tree_samples", metavar="FILE",
	help="List of sample names to create beautiful tree")
parser.add_argument("-maf_save", action="store_true",
	help="Save population allele frequencies")
parser.add_argument("-pi_save", action="store_true",
	help="Save individual allele frequencies")
parser.add_argument("-dosage_save", action="store_true",
	help="Save genotype dosages")
parser.add_argument("-post_save", action="store_true",
	help="Save posterior genotype probabilities")
parser.add_argument("-sites_save", action="store_true",
	help="Save boolean vector of used sites")
parser.add_argument("-threads", metavar="INT", type=int, default=1,
	help="Number of threads")
parser.add_argument("-out", metavar="OUTPUT", default="pcangsd",
	help="Prefix for output files")
args = parser.parse_args()

# Check input
assert (args.beagle is not None) or (args.plink is not None), \
		"Please provide input data (args.beagle or args.plink)!"

# Libraries
import os
import sys
import subprocess
from datetime import datetime

# Find length of PLINK files
def extract_length(filename):
	process = subprocess.Popen(['wc', '-l', filename], stdout=subprocess.PIPE)
	result, err = process.communicate()
	return int(result.split()[0])

# Create log-file of arguments
full = vars(parser.parse_args())
deaf = vars(parser.parse_args([]))
with open(args.out + ".args", "w") as f:
	f.write("PCAngsd v.1.0\n")
	f.write("Time: " + datetime.now().strftime("%d/%m/%Y %H:%M:%S") + "\n")
	f.write("Directory: " + str(os.getcwd()) + "\n")
	f.write("Options:\n")
	for key in full:
		if full[key] != deaf[key]:
			if type(full[key]) is bool:
				f.write("\t-" + str(key) + "\n")
			else:
				f.write("\t-" + str(key) + " " + str(full[key]) + "\n")
del full, deaf

# Control threads
os.environ["OMP_NUM_THREADS"] = str(args.threads)
os.environ["OPENBLAS_NUM_THREADS"] = str(args.threads)
os.environ["MKL_NUM_THREADS"] = str(args.threads)

# Numerical libraries
import numpy as np
from math import ceil

# Import scripts
import reader_cy
import shared
import covariance
import selection
import inbreed
import admixture
import tree

##### PCAngsd #####
print("PCAngsd v.1.01")
print("Using " + str(args.threads) + " thread(s).\n")

# Parse data
if args.beagle is not None:
	print("Parsing Beagle file.")
	if args.filter is None:
		assert os.path.isfile(args.beagle), "Beagle file doesn't exist!"
		L = reader_cy.readBeagle(args.beagle)
		m = L.shape[0]
		n = L.shape[1]//3
	else:
		nFilter = np.repeat(np.genfromtxt(args.filter, dtype=np.uint8), 3)
		print("Only loading " + str(int(np.sum(nFilter)//3)) + "/" + \
				str(nFilter.shape[0]//3) + " individuals.")
		L = reader_cy.readBeagleFilter(args.beagle, nFilter, int(np.sum(nFilter)))
		m = L.shape[0]
		n = L.shape[1]//3
		del nFilter
else:
	print("Parsing PLINK files.")
	assert args.filter is None, "Please perform sample filtering in PLINK!"
	assert os.path.isfile(args.plink + ".bed"), "Bed file doesn't exist!"
	assert os.path.isfile(args.plink + ".bim"), "Bim file doesn't exist!"
	assert os.path.isfile(args.plink + ".fam"), "Fam file doesn't exist!"

	# Count number of sites and samples
	m = extract_length(args.plink + ".bim")
	n = extract_length(args.plink + ".fam")
	with open(args.plink + ".bed", "rb") as bed:
		G = np.fromfile(bed, dtype=np.uint8, offset=3)
	G_len = ceil(n/4)
	G = G.reshape(m, G_len)
	L = np.zeros((m, 3*n), dtype=np.float32)
	reader_cy.convertBed(L, G, G_len, args.plink_error, m, n, args.threads)
print("Loaded " + str(m) + " sites and " + str(n) + " individuals.")
m_old = L.shape[0] # For future reference

### Estimate MAF ###
print("Estimating minor allele frequencies.")
f = shared.emMAF(L, args.maf_iter, args.maf_tole, args.threads)

# Filtering (MAF)
if args.minMaf > 0.0:
	maskMAF = (f >= args.minMaf) & (f <= (1 - args.minMaf))

	# Update arrays
	m = np.sum(maskMAF)
	tmpMask = maskMAF.astype(np.uint8)
	reader_cy.filterArrays(L, f, tmpMask)
	L = L[:m,:]
	f = f[:m]
	del tmpMask
	print("Number of sites after MAF filtering (" + str(args.minMaf) + "): " \
            + str(m))
	m = L.shape[0]

# Filtering (HWE)
if args.hwe is not None:
	lrt = np.load(args.hwe)
	assert f.shape[0] == lrt.shape[0], "Number of LRTs must match the number of sites!"
	from scipy.stats import chi2
	maskHWE = (lrt < chi2.ppf(1 - args.hwe_tole, 1))

	# Update arrays
	m = np.sum(maskHWE)
	tmpMask = maskHWE.astype(np.uint8)
	reader_cy.filterArrays(L, f, tmpMask)
	L = L[:m,:]
	f = f[:m]
	del tmpMask
	print("Number of sites after HWE filtering (" + str(args.hwe_tole) + "): " \
			+ str(np.sum(maskHWE)))
	m = L.shape[0]

### Covariance estimation ###
if args.pi is None:
	print("\nEstimating covariance matrix.")
	C, P, K = covariance.emPCA(L, f, args.e, args.iter, args.tole, args.threads)

	# Save covariance matrix
	np.savetxt(args.out + ".cov", C)
	print("Saved covariance matrix as " + str(args.out) + ".cov (Text).\n")
	del C

	# Exit for standard PCA
	if args.iter == 0:
		sys.exit(0)
else:
	assert args.e != 0, "Must specify number of eigenvectors used!"
	print("Loading pre-estimated frequency matrix.\n")
	P = np.load(args.pi)
	assert P.shape[0] == L.shape[0], "Must have same number of sites as data!"
	K = args.e

### Selection scan and/or SNP weights ###
if args.selection_e != 0:
	s_K = args.selection_e
else:
	s_K = K

# Galinsky scan
if args.selection:
	print("Performing selection scan (FastPCA) for " + str(s_K) + " PCs.")
	D = selection.galinskyScan(L, P, f, s_K, args.threads)

	# Save test statistics
	np.save(args.out + ".selection", D.astype(float))
	print("Saved test statistics as " + str(args.out) + ".selection.npy (Binary).\n")
	del D

# SNP weights
if args.snp_weights:
	print("Estimating SNP weights for " + str(s_K) + " PCs")
	snpW = selection.snpWeights(L, P, f, s_K, args.threads)

	# Save SNP weights
	np.save(args.out + ".weights", snpW.astype(float))
	print("Saved SNP weights as " + str(args.out) + ".weights.npy (Binary).\n")
	del snpW

# pcadapt scan
if args.pcadapt:
	print("Performing selection scan (pcadapt) using " + str(s_K) + " PCs.")
	Zscores = selection.pcadaptScan(L, P, f, s_K, args.threads)

	# Save test statistics
	np.save(args.out + ".pcadapt.zscores", Zscores.astype(float))
	print("Saved z-scores as " + str(args.out) + ".pcadapt.zscores.npy (Binary).")
	print("Use provided script for obtaining p-values (pcadapt.R).\n")
	del Zscores

### HWE - per-site inbreeding coefficients ###
if args.inbreedSites:
	print("Estimating per-site inbreeding cofficients and performing LRT.")
	F, T = inbreed.inbreedSites(L, P, args.inbreed_iter, args.inbreed_tole, \
								args.threads)

	# Save inbreeding coefficients and LRTs
	np.save(args.out + ".inbreed.sites", F.astype(float))
	print("Saved per-site inbreeding coefficients as " + str(args.out) + \
			".inbreed.sites.npy (Binary).")
	np.save(args.out + ".lrt.sites", T.astype(float))
	print("Saved likelihood ratio tests as " + str(args.out) + \
			".lrt.sites.npy (Binary).\n")
	del F, T

### Inbreeding - per-individual inbreeding coefficients ###
if args.inbreedSamples:
	print("Estimating per-individual inbreeding coefficients.")
	F = inbreed.inbreedSamples(L, P, args.inbreed_iter, args.inbreed_tole, \
								args.threads)

	# Save inbreeding coefficients
	np.savetxt(args.out + ".inbreed.samples", F)
	print("Saved per-individual inbreeding coefficients as " + str(args.out) + \
			".inbreed.samples (Text).")
	if args.genoInbreed is None:
		del F

### Genotype calling ###
if args.geno is not None:
	print("Calling genotypes with threshold " + str(args.geno))
	G = shared.callGeno(L, P, None, args.geno, args.threads)

	# Save genotype matrix
	np.save(args.out + ".geno", G)
	print("Saved called genotype matrix as " + str(args.out) + \
			".geno.npy (Binary - np.int8)\n")
	del G

if args.genoInbreed is not None:
	print("Calling genotypes (inbreeding) with threshold " + str(args.genoInbreed))
	G = shared.callGeno(L, P, F, args.genoInbreed, args.threads)

	# Save genotype matrix
	np.save(args.out + ".geno.inbreed", G)
	print("Saved called genotype matrix as " + str(args.out) + \
			".geno.inbreed.npy (Binary - np.int8)\n")
	del G, F

### Admixture estimation ###
if args.admix:
	print("Estimating admixture proportions using NMF (CSG-MU).")
	if args.admix_K is not None:
		a_K = args.admix_K
	else:
		a_K = K + 1
	if args.admix_auto is None:
		print("K=" + str(a_K) + ", Alpha=" + str(args.admix_alpha) + \
				", Batches=" + str(args.admix_batch) + ", Seed=" + str(args.admix_seed))
		Q, F, _ = admixture.admixNMF(L, P, a_K, args.admix_alpha, args.admix_iter, \
										args.admix_tole, args.admix_batch, \
										args.admix_seed, True, args.threads)

		# Save factor matrices
		np.save(args.out + ".admix." + str(a_K) + ".Q", Q.astype(float))
		print("Saved admixture proportions as " + str(args.out) + ".admix." + \
				str(a_K) + ".Q.npy (Binary)")
		np.save(args.out + ".admix." + str(a_K) + ".F", F.astype(float))
		print("Saved ancestral allele frequencies proportions as " + \
				str(args.out) + ".admix." + str(a_K) + ".F.npy (Binary)\n")
	else:
		print("Automatic search for best alpha with depth=" + str(args.admix_depth))
		print("K=" + str(a_K) + ", Batches=" + \
				str(args.admix_batch) + ", Seed=" + str(args.admix_seed))
		Q, F, lB, aB = admixture.alphaSearch(L, P, a_K, args.admix_auto, \
											args.admix_iter, args.admix_tole, \
											args.admix_batch, args.admix_seed, \
											args.admix_depth, args.threads)
		print("Search concluded: Alpha=" + str(aB) + ", log-likelihood" + str(lB))

		# Save factor matrices
		np.save(args.out + ".admix." + str(a_K) + ".Q", Q.astype(float))
		print("Saved admixture proportions as " + str(args.out) + ".admix." + \
				str(a_K) + ".Q.npy (Binary)")
		np.save(args.out + ".admix." + str(a_K) + ".F", F.astype(float))
		print("Saved ancestral allele frequencies proportions as " + \
				str(args.out) + ".admix." + str(a_K) + ".F.npy (Binary)\n")
	del Q, F

### Tree estimation ###
if args.tree:
	if args.tree_samples is not None:
		sList = []
		with open(args.tree_samples, "r") as f_samples:
			for line in f_samples:
				sList.append(line.strip("\n"))
	else:
		sList = [str(i+1) for i in range(n)]
	print("Constructing neighbour-joining tree based on covariance matrix " + \
			"of individual allele frequencies.")
	C = tree.covariancePi(P, f, args.threads)
	newick = tree.constructTree(C, sList)

	# Save tree
	with open(args.out + ".tree", "w") as f_tree:
		f_tree.write(newick)
		f_tree.write(";\n")
	print("Saved newick tree as " + str(args.out) + ".tree (Text)")
	np.savetxt(args.out + ".tree.cov", C)
	print("Saved tree covariance matrix as " + str(args.out) + ".tree.cov (Text)")
	del C, newick

### Optional saves ###
# Minor allele frequencies
if args.maf_save:
	np.save(args.out + ".maf", f.astype(float))
	print("Saved minor allele frequencies as " + str(args.out) + \
			".maf.npy (Binary)\n")

# Posterior expectation of the genotypes (dosages)
if args.dosage_save:
	import covariance_cy
	E = np.zeros(P.shape, dtype=np.float32) # Dosage matrix
	covariance_cy.updateDosages(L, P, E, args.threads)
	np.save(args.out + ".dosage", E)
	print("Saved genotype dosages as " + str(args.out) + \
			".dosage.npy (Binary - np.float32)\n")
	del E

# Posterior genotype probabilities
if args.post_save:
	import shared_cy
	shared_cy.computePost(L, P, args.threads)
	np.save(args.out + ".post.beagle", L)
	print("Saved posterior genotype probabilities as " + str(args.out) + \
			".post.beagle (Binary - np.float32)")

# Individual allele frequencies
if args.pi_save:
	np.save(args.out + ".pi", P)
	print("Saved individual allele frequencies as " + str(args.out) + \
			".pi.npy (Binary - np.float32)\n")
	del P

# Sites "boolean" vector
if args.sites_save:
	siteVec = np.zeros(m_old, dtype=np.uint8)
	if args.minMaf > 0.0:
		if args.hwe is not None:
			siteArr = np.arange(m_old, dtype=int)[maskMAF][maskHWE]
			siteVec[siteArr] = 1
		else:
			siteVec[maskMAF] = 1
	elif args.hwe is not None:
		siteVec[maskHWE] = 1
	else:
		print("All sites have been kept.")
		siteVec[:] = 1
	np.savetxt(args.out + ".sites", siteVec, fmt="%i")
	print("Saved boolean vector of sites kept after filtering as " + \
			str(args.out) + ".sites (Text)")
