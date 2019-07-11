"""
PCAngsd Framework: Population genetic analyses for NGS data using PCA. Main caller.
Python 2/3 compatible based on Cython.
"""

__author__ = "Jonas Meisner"

# Import libraries
import argparse
import numpy as np
import os

# Import functions
import reader
import emMaf
import covariance
import emInbreed
import selection
import kinship
import callGeno
import admixture

##### Argparse #####
parser = argparse.ArgumentParser(prog="PCAngsd")
parser.add_argument("--version", action="version", version="%(prog)s 0.981")
parser.add_argument("-beagle", metavar="FILE",
	help="Input file of genotype likelihoods in gzipped Beagle format from ANGSD")
parser.add_argument("-plink", metavar="FILE-PREFIX",
	help="Prefix for PLINK files (.bed, .bim. .fam)")
parser.add_argument("-plink_error", metavar="FLOAT", type=float, default=0.0,
	help="Incorporate genotype error model")
parser.add_argument("-minMaf", metavar="FLOAT", type=float, default=0.05,
	help="Minimum minor allele frequency threshold (0.05)")
parser.add_argument("-iter", metavar="INT", type=int, default=100,
	help="Maximum iterations for estimation of individual allele frequencies (100)")
parser.add_argument("-tole", metavar="FLOAT", type=float, default=1e-5,
	help="Tolerance for update in estimation of individual allele frequencies (1e-5)")
parser.add_argument("-maf_iter", metavar="INT", type=int, default=200,
	help="Maximum iterations for population allele frequencies estimation - EM (100)")
parser.add_argument("-maf_tole", metavar="FLOAT", type=float, default=1e-4,
	help="Tolerance for population allele frequencies estimation update - EM (1e-4)")
parser.add_argument("-e", metavar="INT", type=int, default=0,
	help="Manual selection of eigenvectors used for SVD")
parser.add_argument("-hwe", metavar="FILE",
	help="Input file of LRTs from HWE test for site filtering")
parser.add_argument("-hwe_tole", metavar="FLOAT", type=float, default=1e-6,
	help="Tolerance for HWE filtering of sites")
parser.add_argument("-selection", action="store_true",
	help="Perform PC-based genome-wide selection scan")
parser.add_argument("-kinship", action="store_true",
	help="Compute kinship matrix adjusted for population structure")
parser.add_argument("-relate", metavar="FILE",
	help="Input kinship matrix to remove related individuals from Beagle file")
parser.add_argument("-relate_tole", metavar="FLOAT", type=float, default=0.0625,
	help="Tolerance for removing related individuals from Beagle file")
parser.add_argument("-inbreedSites", action="store_true",
	help="Compute the per-site inbreeding coefficients and LRT")
parser.add_argument("-inbreed", metavar="INT", type=int,
	help="Compute the per-individual inbreeding coefficients by specified model")
parser.add_argument("-inbreed_iter", metavar="INT", type=int, default=200,
	help="Maximum iterations for inbreeding coefficients estimation - EM (200)")
parser.add_argument("-inbreed_tole", metavar="FLOAT", type=float, default=1e-4,
	help="Tolerance for inbreeding coefficients estimation update - EM (1e-4)")
parser.add_argument("-geno", metavar="FLOAT", type=float,
	help="Call genotypes from posterior probabilities with threshold")
parser.add_argument("-genoInbreed", metavar="FLOAT", type=float,
	help="Call genotypes (inbreeding) from posterior probabilities with threshold")
parser.add_argument("-admix", action="store_true",
	help="Estimate admixture proportions using NMF")
parser.add_argument("-admix_K", metavar="INT", type=int,
	help="Number of ancestral populations in admixture estimation")
parser.add_argument("-admix_iter", metavar="INT", type=int, default=200,
	help="Maximum iterations for admixture estimation - NMF (200)")
parser.add_argument("-admix_tole", metavar="FLOAT", type=float, default=1e-5,
	help="Tolerance for admixture proportions update - NMF (1e-5)")
parser.add_argument("-admix_alpha", metavar="FLOAT-LIST", type=float, nargs="+", 
	default=[0], help="Alpha used for regularization in admixture estimation")
parser.add_argument("-admix_batch", metavar="INT", type=int, default=10,
	help="Number of batches to use in stochastic admixture estimation - NMF (10)")
parser.add_argument("-admix_seed", metavar="INT-LIST", type=int, nargs="+", default=[0],
	help="Random seed used for initialization of factor matrices")
parser.add_argument("-admix_auto", metavar="FLOAT", type=float,
	help="Enable automatic search for alpha by giving soft upper search bound")
parser.add_argument("-admix_depth", metavar="INT", type=int, default=5,
	help="Depth in automatic search of alpha")
parser.add_argument("-admix_save", action="store_true",
	help="Save population-specific allele frequencies (admixture)")
parser.add_argument("-maf_save", action="store_true",
	help="Save population allele frequencies")
parser.add_argument("-indf_save", action="store_true",
	help="Save individual allele frequencies")
parser.add_argument("-dosage_save", action="store_true",
	help="Save genotype dosages")
parser.add_argument("-sites_save", action="store_true",
	help="Save site IDs")
parser.add_argument("-post_save", action="store_true",
	help="Save posterior genotype probabilities")
parser.add_argument("-threads", metavar="INT", type=int, default=1,
	help="Number of threads")
parser.add_argument("-o", metavar="OUTPUT", help="Prefix output file name", default="pcangsd")
args = parser.parse_args()

print("PCAngsd 0.981")
print("Using " + str(args.threads) + " thread(s)\n")

# Individual filtering based on relatedness
if args.relate is not None:
	phi = np.load(args.relate)

	# Boolean vector of unrelated individuals
	print("Masking related individuals with pair-wise kinship estimates >= " + str(args.relate_tole))
	phi[np.triu_indices(phi.shape[0])] = 0 # Setting half of matrix to 0
	if len(np.unique(np.where(phi > args.relate_tole)[0])) < len(np.unique(np.where(phi > args.relate_tole)[1])):
		relatedIndices = np.unique(np.where(phi > args.relate_tole)[0])
	else:
		relatedIndices = np.unique(np.where(phi > args.relate_tole)[1])
	unrelatedI = np.isin(np.arange(phi.shape[0]), relatedIndices, invert=True)
	
	# Create or update index vector
	n = sum(unrelatedI)
	np.save(args.o + ".unrelated", unrelatedI.astype(int))
	print("Keeping " + str(n) + " individuals after filtering (removing " + str(phi.shape[0] - n) + ")")
	print("Boolean vector of unrelated individuals saved as " + str(args.o) + ".unrelated.npy (Binary)")
	unrelatedI = np.repeat(unrelatedI, 3)
	del phi, relatedIndices

# Parse input file
if args.beagle is not None:
	print("Parsing Beagle file")
	assert os.path.isfile(args.beagle), "Beagle file doesn't exist!"
	assert args.beagle[-3:] == ".gz", "Beagle file must be in gzip format!"
	if args.relate is None:
		L = reader.readBeagle(args.beagle).T
	else:
		L = reader.readBeagleUnrelated(args.beagle, unrelatedI).T
		del unrelatedI
elif args.plink is not None:
	print("Parsing PLINK files")
	assert os.path.isfile(args.plink + ".bed"), "Bed file doesn't exist"
	from pandas_plink import read_plink
	import shared
	bim, fam, bed = read_plink(args.plink, verbose=False)
	L = np.empty((3*bed.shape[1], bed.shape[0]), dtype=np.float32)
	shared.convertBed(L, bed.T.compute(), args.plink_error, args.threads)
	del bed

n, m = L.shape
n //= 3
print("Read " + str(n) + " samples and " + str(m) + " sites\n")

##### Estimate population allele frequencies #####
print("Estimating population allele frequencies")
f = emMaf.alleleEM(L, args.maf_iter, args.maf_tole, args.threads)

# Filtering (MAF)
if args.minMaf > 0.0:
	maskMAF = (f >= args.minMaf) & (f <= 1 - args.minMaf)
	
	# Update arrays
	f = np.compress(maskMAF, f)
	L = np.compress(maskMAF, L, axis=1)
	print("Number of sites after MAF filtering (" + str(args.minMaf) + "): " + str(np.sum(maskMAF)) + "\n")

# Filtering (HWE)
if args.hwe is not None:
	lrt = np.load(args.hwe)
	assert f.shape[0] == lrt.shape[0], "Number of LRTs must match the number of sites!"
	import scipy.stats as st
	threshold = st.chi2.ppf(1 - args.hwe_tole, 1)
	maskHWE = (lrt < threshold)

	# Update arrays
	f = np.compress(maskHWE, f)
	L = np.compress(maskHWE, L, axis=1)
	print("Number of sites after HWE filtering (" + str(args.hwe_tole) + "): " + str(np.sum(maskHWE)) + "\n")

##### PCAngsd - Individual allele frequencies and covariance matrix #####
print("Estimating covariance matrix")
C, Pi, e = covariance.pcaEM(L, args.e, f, args.iter, args.tole, args.threads)

# Save covariance matrix
np.savetxt(args.o + ".cov", C)
print("Saved covariance matrix as " + str(args.o) + ".cov (Text)\n")
del C

##### Selection scan
if args.selection:
	print("Performing PC-based selection scan")
	Dsquared = selection.selectionScan(L, Pi, f, e, args.threads)

	# Save test statistics
	np.save(args.o + ".selection", Dsquared.astype(float))
	print("Saved test statistics as " + str(args.o) + ".selection.npy (Binary)\n")

	# Release memory
	del Dsquared

##### Kinship
if args.kinship:
	print("Estimating kinship matrix")
	phi = kinship.kinshipConomos(L, Pi, args.threads)

	# Save kinship matrix
	np.save(args.o + ".kinship", phi.astype(float))
	print("Saved kinship matrix as " + str(args.o) + ".kinship.npy (Binary)\n")

	if args.inbreed is None:
		# Release memory
		del phi

##### Per-site inbreeding coefficients
if args.inbreedSites:
	print("Estimating per-site inbreeding coefficients and performing LRT")
	Fsites, lrt = emInbreed.inbreedSitesEM(L, Pi, args.inbreed_iter, args.inbreed_tole, args.threads)

	# Save inbreeding coefficients and LRTs
	np.save(args.o + ".inbreed.sites", Fsites.astype(float))
	print("Saved per-site inbreeding coefficients as " + str(args.o) + ".inbreed.sites.npy (Binary)")
	np.save(args.o + ".lrt.sites", lrt.astype(float))
	print("Saved likelihood ratio tests as " + str(args.o) + ".lrt.sites.npy (Binary)\n")

	# Release memory
	del Fsites, lrt

##### Per-individual inbreeding coefficients
if args.inbreed is not None:
	assert args.inbreed in [1,2,3], "Invalid model chosen for per-individual inbreeding coefficients!"
	print("Estimating per-individual inbreeding coefficients")
	if args.inbreed in [1,2]:
		Find = emInbreed.inbreedEM(L, Pi, args.inbreed, args.inbreed_iter, args.inbreed_tole, args.threads)
	else:
		print("Using Kinship coefficients")
		Find = 2*np.diag(phi) - 1

	# Save inbreeding coefficients
	np.save(args.o + ".inbreed", Find.astype(float))
	print("Saved per-individual inbreeding coefficients as " + str(args.o) + ".inbreed.npy (Binary)\n")

	if args.genoInbreed is None:
		# Release memory
		del Find

##### Genotype calling
if args.geno is not None:
	print("Calling genotypes with threshold " + str(args.geno))
	G = callGeno.callGeno(L, Pi, None, args.geno, args.threads)

	# Save genotype matrix
	np.save(args.o + ".geno", G)
	print("Saved called genotype matrix as " + str(args.o) + ".geno.npy (Binary)\n")

	# Release memory
	del G

if args.genoInbreed is not None:
	assert args.inbreed is not None, "Must have estimated per-individual inbreeding coefficients!"
	print("Calling genotypes (inbreeding) with threshold " + str(args.genoInbreed))
	G = callGeno.callGeno(L, Pi, Find, args.genoInbreed, args.threads)

	# Save genotype matrix
	np.save(args.o + ".geno.inbreed", G)
	print("Saved called genotype matrix as " + str(args.o) + ".geno.inbreed.npy (Binary)\n")

	# Release memory
	del G, Find

##### Admixture
if args.admix:
	if args.admix_K is None:
		K = e + 1
	else:
		K = args.admix_K

	if args.admix_auto is None:
		if (len(args.admix_alpha) > 1) or (len(args.admix_seed) > 1):
			for aAlpha in args.admix_alpha:
				for aSeed in args.admix_seed:
					print("Estimating admixture with K=" + str(K) + ", alpha=" + str(aAlpha) \
						+ ", batch=" + str(args.admix_batch) + ", seed=" + str(aSeed))
					Q, F, _ = admixture.admixNMF(L, Pi, K, aAlpha, args.admix_iter, args.admix_tole, \
						aSeed, args.admix_batch, True, args.threads)

					# Save factor matrices
					np.save(args.o + ".admix.Q.a" + str(aAlpha) + ".s" + str(aSeed), Q)
					print("Saved admixture proportions as " + str(args.o) + ".admix.Q.a" + str(aAlpha) \
						+ ".s" + str(aSeed) + ".npy (Binary)")
					if args.admix_save:
						np.save(args.o + ".admix.F.a" + str(aAlpha) + ".s" + str(aSeed), F.T.astype(float))
						print("Saved ancestral allele frequencies as " + str(args.o) + ".admix.F.a" \
							+ str(aAlpha) + ".s" + str(aSeed) + ".npy (Binary)")
					print("\n")
		else:
			print("Estimating admixture with K=" + str(K) + ", alpha=" + str(args.admix_alpha[0]) \
				+ ", batch=" + str(args.admix_batch) + ", seed=" + str(args.admix_seed[0]))
			Q, F, _ = admixture.admixNMF(L, Pi, K, args.admix_alpha[0], args.admix_iter, args.admix_tole, \
				args.admix_seed[0], args.admix_batch, True, args.threads)

			# Save factor matrices
			np.save(args.o + ".admix.Q", Q.astype(float))
			print("Saved admixture proportions as " + str(args.o) + ".admix.Q.npy (Binary)")
			if args.admix_save:
				np.save(args.o + ".admix.F", F.T.astype(float))
				print("Saved ancestral allele frequencies as " + str(args.o) + ".admix.F.npy (Binary)")
			print("\n")
	else:
		print("Automatic alpha selection in admixture estimation. K=" + str(K) + ", Depth=" + str(args.admix_depth) \
			+ ", batch=" + str(args.admix_batch) + ", seed=" + str(args.admix_seed[0]))
		Q, F, L_best, aBest = admixture.alphaSearch(L, Pi, K, args.admix_auto, args.admix_iter, args.admix_tole, \
			args.admix_seed[0], args.admix_batch, args.admix_depth, args.threads)
		print("\nAutomatic search concluded: Alpha=" + str(aBest) + ", Log-likelihood=" + str(L_best))

		# Save factor matrices
		np.save(args.o + ".admix.Q", Q.astype(float))
		print("Saved admixture proportions as " + str(args.o) + ".admix.Q.npy (Binary)")
		if args.admix_save:
			np.save(args.o + ".admix.F", F.T.astype(float))
			print("Saved ancestral allele frequencies as " + str(args.o) + ".admix.F.npy (Binary)")
		print("\n")

	# Release memory
	del Q, F

##### Optional saves
def writeBeagle(L, infoDF, infoHeader):
	_, m = L.shape
	with open(args.o + ".post.beagle", "w") as f:
		f.write("\t".join(infoHeader) + "\n")
		for j in range(m):
			f.write("\t".join(list(infoDF.iloc[j,:])) + "\t")
			L[:, j].tofile(f, sep="\t", format="%.6f")
			f.write("\n")

if args.maf_save:
	np.save(args.o + ".maf", f.astype(float))
	print("Saved population allele frequencies as " + str(args.o) + ".maf.npy (Binary)\n")

if args.indf_save:
	np.save(args.o + ".indf", Pi.astype(float))
	print("Saved individual allele frequencies as " + str(args.o) + ".indf.npy (Binary)\n")

if args.dosage_save:
	import covariance_cy
	E = np.empty((L.shape[0]//3, L.shape[1]), dtype=np.float32) # Dosage matrix
	covariance_cy.updatePCAngsd(L, Pi, E, args.threads)
	np.save(args.o + ".dosage", E.astype(float))
	print("Saved genotype dosages as " + str(args.o) + ".dosage.npy (Binary)\n")
	del E

if args.post_save or args.sites_save:
	print("Loading site information")
	import pandas as pd
	if args.beagle is not None:
		infoDF = pd.read_csv(args.beagle, sep="\t", header=0, usecols=[0,1,2], compression="gzip", dtype=str)
	else:
		siteS = bim["chrom"].astype(str)+'_'+bim["pos"].astype(str)
		infoDF = pd.DataFrame(dict(marker=siteS, allele1=bim["a1"], allele2=bim["a0"]), dtype=str)
	if args.minMaf > 0.0:
		infoDF = infoDF[maskMAF]
	if args.hwe is not None:
		infoDF = infoDF[maskHWE]
	infoDF.reset_index(drop=True)
	if args.sites_save:
		infoDF.iloc[:,0].to_csv(args.o + ".sites", header=False, index=False)
		print("Saved site IDs as " + str(args.o) + ".sites (Text)")
	if args.post_save:
		import shared
		if args.iter != 0:
			print("Saving posterior genotype probabilities (PCAngsd)")
			shared.computePostPi(L, Pi, args.threads)
		else:
			print("Saving posterior genotype probabilities (No population structure)")
			shared.computePostF(L, f, args.threads)
		if args.beagle is not None:
			infoHeader = ["marker", "allele1", "allele2"] + ["Ind" + str(i) for j in range(L.shape[0]//3) for i in [j, j, j]]
		else:
			infoHeader = ["marker", "allele1", "allele2"] + [fam["iid"][j] for j in range(L.shape[0]//3) for i in [j, j, j]]
		writeBeagle(L, infoDF, infoHeader)
		print("Saved posterior genotype probabilities as " + str(args.o) + ".post.beagle (Text)")