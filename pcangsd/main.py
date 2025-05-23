"""
PCAngsd.
Main caller.

Jonas Meisner and Anders Albrechtsen
"""

__author__ = "Jonas Meisner"

# Libraries
import argparse
import os
import sys
from datetime import datetime
from time import time

VERSION = "1.36.3"

# Argparse
parser = argparse.ArgumentParser(prog="pcangsd")
parser.add_argument("--version", action="version",
	version=f"v{VERSION}")
parser.add_argument("-b", "--beagle", metavar="FILE",
	help="Filepath to genotype likelihoods in gzipped Beagle format from ANGSD")
parser.add_argument("-p", "--plink", metavar="FILE-PREFIX",
	help="Prefix PLINK files (.bed, .bim, .fam)")
parser.add_argument("-e", "--eig", metavar="INT", type=int, default=0,
	help="Manual selection of number of eigenvectors in modelling")
parser.add_argument("-t", "--threads", metavar="INT", type=int, default=1,
	help="Number of threads (1)")
parser.add_argument("-o", "--out", metavar="OUTPUT", default="pcangsd",
	help="Prefix for output files")
parser.add_argument("--filter", metavar="FILE",
	help="Input file of vector for filtering samples")
parser.add_argument("--filter-sites", metavar="FILE",
	help="Input file of vector for filtering sites")
parser.add_argument("--geno-error", metavar="FLOAT", type=float, default=0.0,
	help="Incorporate errors into genotypes (0.0)")
parser.add_argument("--maf", metavar="FLOAT", type=float, default=0.05,
	help="Minimum minor allele frequency threshold (0.05)")
parser.add_argument("--maf-iter", metavar="INT", type=int, default=500,
	help="Maximum iterations for minor allele frequencies estimation - EM (500)")
parser.add_argument("--maf-tole", metavar="FLOAT", type=float, default=1e-6,
	help="Tolerance for minor allele frequencies estimation update - EM (1e-6)")
parser.add_argument("--hwe", metavar="FILE",
	help="Input file of LRTs from HWE test for site filtering")
parser.add_argument("--hwe-tole", metavar="FLOAT", type=float, default=1e-6,
	help="Tolerance for HWE filtering of sites (1e-6)")
parser.add_argument("--iter", metavar="INT", type=int, default=100,
	help="Maximum iterations for estimation of individual allele frequencies (100)")
parser.add_argument("--tole", metavar="FLOAT", type=float, default=1e-5,
	help="Tolerance for update in estimation of individual allele frequencies (1e-5)")
parser.add_argument("--selection", action="store_true",
	help="Perform PC-based genome-wide selection scan")
parser.add_argument("--snp-weights", action="store_true",
	help="Estimate SNP weights")
parser.add_argument("--pcadapt", action="store_true",
	help="Perform pcadapt selection scan")
parser.add_argument("--selection-eig", metavar="INT", type=int, default=0,
	help="Manual selection of number of eigenvectors in selection scans/SNP weights")
parser.add_argument("--inbreed-sites", action="store_true",
	help="Compute the per-site inbreeding coefficients and LRT")
parser.add_argument("--inbreed-samples", action="store_true",
	help="Compute the per-sample inbreeding coefficients")
parser.add_argument("--inbreed-iter", metavar="INT", type=int, default=500,
	help="Maximum iterations for inbreeding coefficients estimation - EM (500)")
parser.add_argument("--inbreed-tole", metavar="FLOAT", type=float, default=1e-5,
	help="Tolerance for inbreeding coefficients estimation update - EM (1e-5)")
parser.add_argument("--post", action="store_true",
	help="Save genotype posteriors")
parser.add_argument("--post-inbreed", action="store_true",
	help="Save genotype posteriors (inbreeding)")
parser.add_argument("--geno", metavar="FLOAT", type=float,
	help="Call genotypes from posterior probabilities with threshold")
parser.add_argument("--geno-inbreed", metavar="FLOAT", type=float,
	help="Call genotypes (inbreeding) from posterior probabilities with threshold")
parser.add_argument("--admix", action="store_true",
	help="Estimate admixture proportions and ancestral allele frequencies")
parser.add_argument("--admix-K", metavar="INT", type=int,
	help="Number of admixture components")
parser.add_argument("--admix-iter", metavar="INT", type=int, default=500,
	help="Maximum number of iterations for admixture estimations - NMF (500)")
parser.add_argument("--admix-tole", metavar="FLOAT", type=float, default=1e-5,
	help="Tolerance for admixture estimations - NMF (1e-5)")
parser.add_argument("--admix-batch", metavar="INT", type=int, default=10,
	help="Number of mini-batches in stochastic admixture estimations - NMF (10)")
parser.add_argument("--admix-alpha", metavar="FLOAT", type=float, default=0.0,
	help="Alpha value for regularization in admixture estimations - NMF (0.0)")
parser.add_argument("--admix-seed", metavar="INT", type=int, default=0,
	help="Random seed used in admixture estimations - NMF (0)")
parser.add_argument("--admix-auto", metavar="FLOAT", type=float,
	help="Enable automatic search for alpha by giving soft upper search bound")
parser.add_argument("--admix-depth", metavar="INT", type=int, default=7,
	help="Depth in automatic search of alpha (7)")
parser.add_argument("--tree", action="store_true",
	help="Construct NJ tree from covariance matrix")
parser.add_argument("--tree-samples", metavar="FILE",
	help="List of sample names to create beautiful tree")
parser.add_argument("--maf-save", action="store_true",
	help="Save population allele frequencies")
parser.add_argument("--pi-save", action="store_true",
	help="Save individual allele frequencies")
parser.add_argument("--dosage-save", action="store_true",
	help="Save genotype dosages")
parser.add_argument("--sites-save", action="store_true",
	help="Save boolean vector of used sites")



##### PCAngsd #####
def main():
	args = parser.parse_args()
	if len(sys.argv) < 2:
		parser.print_help()
		sys.exit()
	print("------------------------------------")
	print(f"PCAngsd v{VERSION}")
	print("Jonas Meisner and Anders Albrechtsen")
	print(f"Using {args.threads} thread(s)")
	print("------------------------------------\n")

	# Check input
	assert (args.beagle is not None) or (args.plink is not None), \
			"No input data (--beagle or --plink)!"
	assert args.eig >= 0, "Please select a valid number of eigenvectors!"
	assert args.threads > 0, "Please select a valid number of threads!"
	assert args.geno_error >= 0.0, "Please select a valid threshold!"
	assert args.maf >= 0.0, "Please select a valid threshold for MAF!"
	assert args.maf_iter > 0, "Please select a valid number of iterations!"
	assert args.maf_tole >= 0.0, "Please select a valid tolerance!"
	assert args.hwe_tole >= 0.0, "Please select a valid tolerance!"
	assert args.iter > 0, "Please select a valid number of iterations!"
	assert args.tole >= 0.0, "Please select a valid tolerance!"
	assert args.inbreed_iter > 0, "Please select a valid number of iterations!"
	assert args.inbreed_tole >= 0.0, "Please select a valid tolerance!"
	assert args.selection_eig >= 0, "Please select a valid number of eigenvectors!"
	assert args.admix_iter > 0, "Please select a valid number of iterations!"
	assert args.admix_tole >= 0.0, "Please select a valid tolerance!"
	assert args.admix_batch > 0, "Please select a valid number of mini-batches!"
	assert args.admix_alpha >= 0.0, "Please select a valid threshold!"
	assert args.admix_seed >= 0.0, "Please select a valid random seed!"
	assert args.admix_depth > 0, "Please select a valid depth value!"
	if args.post_inbreed or (args.geno_inbreed is not None):
		assert args.inbreed_samples, "Must estimate toggle '--inbreed-samples'!"
	if args.geno is not None:
		assert args.geno >= 0.0, "Please select a valid threshold!"
	if args.geno_inbreed is not None:
		assert args.geno_inbreed >= 0.0, "Please select a valid threshold!"
	if args.admix_K is not None:
		assert args.admix_K >= 2, "Please select a valid number of components!"
	if args.admix_auto is not None:
		assert args.admix_auto > 0.0, "Please select a valid value!"
	if args.filter is not None:
		assert os.path.isfile(args.filter), "Sample filter doesn't exist!"
	if args.filter_sites is not None:
		assert os.path.isfile(args.filter_sites), "Site filter doesn't exist!"
	if args.hwe is not None:
		assert os.path.isfile(args.hwe), "HWE filter file doesn't exist!"
	if args.tree_samples is not None:
		assert os.path.isfile(args.tree_samples), "Sample file doesn't exist!"
	start = time()

	# Create log-file of arguments
	full = vars(parser.parse_args())
	deaf = vars(parser.parse_args([]))
	with open(args.out + ".log", "w") as log:
		log.write(f"PCAngsd v{VERSION}\n")
		log.write(f"Time: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n")
		log.write(f"Directory: {os.getcwd()}\n")
		log.write("Options:\n")
		for key in full:
			if full[key] != deaf[key]:
				if type(full[key]) is bool:
					log.write(f"\t--{key}\n")
				else:
					log.write(f"\t--{key} {full[key]}\n")
	del full, deaf

	# Control threads of external numerical libraries
	os.environ["MKL_NUM_THREADS"] = str(args.threads)
	os.environ["MKL_MAX_THREADS"] = str(args.threads)
	os.environ["OMP_NUM_THREADS"] = str(args.threads)
	os.environ["OMP_MAX_THREADS"] = str(args.threads)
	os.environ["NUMEXPR_NUM_THREADS"] = str(args.threads)
	os.environ["NUMEXPR_MAX_THREADS"] = str(args.threads)
	os.environ["OPENBLAS_NUM_THREADS"] = str(args.threads)
	os.environ["OPENBLAS_MAX_THREADS"] = str(args.threads)

	# Numerical libraries
	import numpy as np
	from pcangsd import admixture
	from pcangsd import covariance
	from pcangsd import inbreed
	from pcangsd import reader_cy
	from pcangsd import selection
	from pcangsd import shared
	from pcangsd import tree

	# Parse data
	if args.beagle is not None:
		print("Parsing Beagle file.")
		if (args.filter is None) and (args.filter_sites is None):
			assert os.path.isfile(args.beagle), "Beagle file doesn't exist!"
			L = reader_cy.readBeagle(args.beagle)
			M = L.shape[0]
			N = L.shape[1]//2
		else:
			if args.filter_sites is not None:
				M_f = np.loadtxt(args.filter_sites, dtype=np.uint8)
				print(f"Only loading {int(np.sum(M_f))}/{M_f.shape[0]} sites.")
				if args.filter is None:
					L = reader_cy.readBeagleFilterSites(args.beagle, M_f)
					M = L.shape[0]
					N = L.shape[1]//2
					if not args.sites_save:
						del M_f
			if args.filter is not None:
				N_f = np.repeat(np.loadtxt(args.filter, dtype=np.uint8), 3)
				print(f"Only loading {int(np.sum(N_f)//3)}/{N_f.shape[0]//3} samples.")
				if args.filter_sites is None:
					L = reader_cy.readBeagleFilterInd(args.beagle, N_f, \
						int(np.sum(N_f)))
					M = L.shape[0]
					N = L.shape[1]//2
					del N_f
			if (args.filter_sites is not None) and (args.filter is not None):
				L = reader_cy.readBeagleFilter(args.beagle, M_f, N_f, \
					int(np.sum(N_f)))
				M = L.shape[0]
				N = L.shape[1]//2
				del N_f
				if not args.sites_save:
					del M_f
	else:
		print("Parsing PLINK files.")
		assert args.filter is None, "Please perform sample filtering in PLINK!"
		assert args.filter_sites is None, "Please perform site filtering in PLINK!"
		assert os.path.isfile(f"{args.plink}.bed"), "bed file doesn't exist!"
		assert os.path.isfile(f"{args.plink}.bim"), "bim file doesn't exist!"
		assert os.path.isfile(f"{args.plink}.fam"), "fam file doesn't exist!"

		# Count number of sites and samples
		L, M, N = shared.readPlink(args.plink, args.geno_error)
	print(f"Loaded {M} sites and {N} samples.")
	if (args.filter_sites is not None) and (args.sites_save):
		M_old = M_f.shape[0]
	else:
		M_old = L.shape[0] # For future reference
	
	# Log data info
	with open(args.out + ".log", "a") as log:
		log.write(f"\nLoaded {M} sites and {N} samples.\n")

	### Estimate MAF
	print("Estimating minor allele frequencies.")
	f = shared.emMAF(L, args.maf_iter, args.maf_tole)

	# Filtering (MAF)
	if args.maf > 0.0:
		assert args.maf < 1.0, "Please provide a valid MAF threshold!"
		maskMAF = (f >= args.maf) & (f <= (1 - args.maf))

		# Update arrays
		M = np.sum(maskMAF, dtype=int)
		tmpMask = maskMAF.astype(np.uint8)
		reader_cy.filterArrays(L, f, tmpMask)
		L = L[:M,:]
		f = f[:M]
		del tmpMask
		print(f"Number of sites after MAF filtering ({args.maf}): {M}")
		M = L.shape[0]

		# Log data info
		with open(args.out + ".log", "a") as log:
			log.write(f"Number of sites after MAF filtering ({args.maf}): {M}\n")

	# Filtering (HWE)
	if args.hwe is not None:
		lrt = np.loadtxt(args.hwe, dtype=float)
		assert f.shape[0] == lrt.shape[0], "Number of LRTs must match number of sites!"
		from scipy.stats import chi2
		maskHWE = (lrt < chi2.ppf(1 - args.hwe_tole, 1))

		# Update arrays
		M = int(np.sum(maskHWE))
		tmpMask = maskHWE.astype(np.uint8)
		reader_cy.filterArrays(L, f, tmpMask)
		L = L[:M,:]
		f = f[:M]
		del tmpMask
		print(f"Number of sites after HWE filtering ({args.hwe_tole}): {M}")
		M = L.shape[0]

		# Log data info
		with open(args.out + ".log", "a") as log:
			log.write(f"Number of sites after HWE filtering ({args.hwe_tole}): {M}\n")

	### Covariance estimation
	print("\nEstimating covariance matrix.")
	C, P, K, it, converged = covariance.emPCA(L, f, args.eig, args.iter, args.tole)

	# Save covariance matrix
	np.savetxt(f"{args.out}.cov", C, fmt="%.7f")
	print(f"Saved covariance matrix as {args.out}.cov\n")
	del C

	# Exit for standard PCA
	if args.iter == 0:
		# Allow for educational purposes to use ngsF inbreeding estimation model
		if args.inbreed_sites:
			print("Estimating per-site inbreeding cofficients and LRT (ngsF).")
			P = shared.fakeFreqs(f, M, N)
			F, T = inbreed.inbreedSites(L, P, args.inbreed_iter, args.inbreed_tole)
			
			# Save inbreeding coefficients and LRTs
			np.savetxt(f"{args.out}.ngsf.inbreed.sites", F, fmt="%.7f")
			print("Saved per-site inbreeding coefficients as " + \
				f"{args.out}.ngsf.inbreed.sites")
			np.savetxt(args.out + ".ngsf.lrt.sites", T, fmt="%.7f")
			print("Saved likelihood ratio tests as " + \
				f"{args.out}.ngsf.lrt.sites\n")
		if args.inbreed_samples:
			print("Estimating per-sample inbreeding coefficients (ngsF).")
			P = shared.fakeFreqs(f, M, N)
			F = inbreed.inbreedSamples(L, P, args.inbreed_iter, args.inbreed_tole)

			# Save inbreeding coefficients
			np.savetxt(f"{args.out}.ngsf.inbreed.samples", F, fmt="%.7f")
			print("Saved per-sample inbreeding coefficients as " + \
				f"{args.out}.inbreed.ngsf.samples\n")

		# Write output info to log-file
		with open(args.out + ".log", "a") as log:
			log.write("\nPCAngsd was run without the iterative process (ngsTools).\n")
			log.write(f"Saved covariance matrix as {args.out}.cov\n")
			if args.inbreed_sites:
				log.write("Saved per-site inbreeding coefficients as " + \
					f"{args.out}.ngsf.inbreed.sites\n")
				log.write("Saved likelihood ratio tests as " + \
					f"{args.out}.ngsf.lrt.sites\n")
			if args.inbreed_samples:
				log.write("Saved per-sample inbreeding coefficients as " + \
					f"{args.out}.inbreed.ngsf.samples\n")
		sys.exit(0)

	### Selection scan and/or SNP weights
	if args.selection_eig != 0:
		s_K = args.selection_eig
	else:
		s_K = K

	# Galinsky scan
	if args.selection:
		print(f"Performing selection scan (FastPCA) for {s_K} PCs.")
		D = selection.galinskyScan(L, P, f, s_K)

		# Save test statistics
		np.savetxt(f"{args.out}.selection", D, fmt="%.7f")
		print(f"Saved test statistics as {args.out}.selection\n")
		del D

	# SNP weights
	if args.snp_weights:
		print(f"Estimating SNP weights for {s_K} PCs.")
		snpW = selection.snpWeights(L, P, f, s_K)

		# Save SNP weights
		np.savetxt(f"{args.out}.weights", snpW, fmt="%.7f")
		print(f"Saved SNP weights as {args.out}.weights\n")
		del snpW

	# pcadapt scan
	if args.pcadapt:
		print(f"Performing selection scan (pcadapt) using {s_K} PCs.")
		Zscores = selection.pcadaptScan(L, P, f, s_K)

		# Save test statistics
		np.savetxt(f"{args.out}.pcadapt.zscores", Zscores, fmt="%.7f")
		print(f"Saved z-scores as {args.out}.pcadapt.zscores")
		print("Use provided script for obtaining p-values (pcadapt.R).\n")
		del Zscores

	### HWE - per-site inbreeding coefficients
	if args.inbreed_sites:
		print("Estimating per-site inbreeding cofficients and performing LRT.")
		F, T = inbreed.inbreedSites(L, P, args.inbreed_iter, args.inbreed_tole)

		# Save inbreeding coefficients and LRTs
		np.savetxt(f"{args.out}.inbreed.sites", F, fmt="%.7f")
		print(f"Saved per-site inbreeding coefficients as {args.out}.inbreed.sites")
		np.savetxt(f"{args.out}.lrt.sites", T, fmt="%.7f")
		print(f"Saved likelihood ratio tests as {args.out}.lrt.sites\n")
		del F, T

	### Inbreeding - per-sample inbreeding coefficients
	if args.inbreed_samples:
		print("Estimating per-sample inbreeding coefficients.")
		F = inbreed.inbreedSamples(L, P, args.inbreed_iter, args.inbreed_tole)

		# Save inbreeding coefficients
		np.savetxt(f"{args.out}.inbreed.samples", F, fmt="%.7f")
		print("Saved per-sample inbreeding coefficients as " + \
			f"{args.out}.inbreed.samples\n")
		if (args.geno_inbreed is None) and (not args.post_inbreed):
			del F

	### Genotype posteriors
	if args.post:
		print(f"Estimating genotype posteriors.")
		G = shared.estimatePost(L, P, None)

		# Save genotype posteriors to text-file
		reader_cy.writeBeagle(G, f"{args.out}.post")
		print(f"Saved genotype posteriors as {args.out}.post\n")
		del G
	
	if args.post_inbreed:
		print(f"Estimating genotype posteriors (inbreeding).")
		G = shared.estimatePost(L, P, F)

		# Save genotype posteriors to text-file
		reader_cy.writeBeagle(G, f"{args.out}.post.inbreed")
		print(f"Saved genotype posteriors as {args.out}.post.inbreed\n")
		del G

	### Genotype calling
	if args.geno is not None:
		print(f"Calling genotypes with threshold {args.geno}.")
		G = shared.callGeno(L, P, None, args.geno)

		# Save genotype matrix
		np.save(f"{args.out}.geno", G)
		print(f"Saved called genotype matrix as {args.out}.geno.npy (Binary)\n")
		del G

	if args.geno_inbreed is not None:
		print(f"Calling genotypes (inbreeding) with threshold {args.geno_inbreed}.")
		G = shared.callGeno(L, P, F, args.geno_inbreed)

		# Save genotype matrix
		np.save(f"{args.out}.geno.inbreed", G)
		print("Saved called genotype matrix as " + \
			f"{args.out}.geno.inbreed.npy (Binary)\n")
		del G, F

	### Tree estimation
	if args.tree:
		if args.tree_samples is not None:
			sList = []
			with open(args.tree_samples, "r") as f_samples:
				for line in f_samples:
					sList.append(line.strip("\n"))
		else:
			sList = [str(i+1) for i in range(N)]
		print("Constructing neighbour-joining tree based on covariance matrix " + \
			"of individual allele frequencies.")
		C = tree.covariancePi(P, f)
		newick = tree.constructTree(C, sList)

		# Save tree
		with open(f"{args.out}.tree", "w") as f_tree:
			f_tree.write(newick)
			f_tree.write(";\n")
		print(f"Saved newick tree as {args.out}.tree")
		np.savetxt(f"{args.out}.tree.cov", C)
		print(f"Saved tree covariance matrix as {args.out}.tree.cov\n")
		del C, newick

	### Admixture estimation
	if args.admix:
		print("Estimating admixture proportions using NMF (CSG-MU).")
		rng = np.random.default_rng(args.admix_seed)
		if args.admix_K is not None:
			a_K = args.admix_K
		else:
			a_K = K + 1
		if args.admix_auto is None:
			print(f"K={a_K}, Alpha={args.admix_alpha}, Batches={args.admix_batch}, " + \
				f"Seed={args.admix_seed}")
			Q, F, _ = admixture.admixNMF(L, P, a_K, args.admix_alpha, args.admix_iter, \
				args.admix_tole, args.admix_batch, rng, True)
		else:
			print(f"Automatic search for best alpha with depth={args.admix_depth}")
			print(f"K={a_K}, Batches={args.admix_batch}, Seed={args.admix_seed}")
			Q, F, lB, aB = admixture.alphaSearch(L, P, a_K, args.admix_auto, \
				args.admix_iter, args.admix_tole, args.admix_batch, rng, \
				args.admix_depth)
			print(f"Search concluded: Alpha={aB}, log-likelihood={lB}")

		# Save factor matrices
		np.savetxt(f"{args.out}.admix.{a_K}.Q", Q, fmt="%.7f")
		print(f"Saved admixture proportions as {args.out}.admix.{a_K}.Q")
		np.savetxt(f"{args.out}.admix.{a_K}.P", F, fmt="%.7f")
		print("Saved ancestral allele frequencies proportions as " + \
			f"{args.out}.admix.{a_K}.P\n")
		del Q, F

	### Optional saves
	# Minor allele frequencies
	if args.maf_save:
		np.savetxt(f"{args.out}.freqs", f, fmt="%.7f")
		print(f"Saved minor allele frequencies as {args.out}.freqs\n")

	# Posterior expectation of the genotypes (dosages)
	if args.dosage_save:
		from pcangsd import covariance_cy
		E = np.zeros(P.shape, dtype=np.float32) # Dosage matrix
		covariance_cy.updateDosages(L, P, E)
		np.save(f"{args.out}.dosage", E)
		print(f"Saved genotype dosages as {args.out}.dosage.npy (Binary)\n")
		del E

	# Individual allele frequencies
	if args.pi_save:
		np.save(f"{args.out}.pi", P)
		print(f"Saved individual allele frequencies as {args.out}.pi.npy (Binary)\n")
		del P

	# Sites "boolean" vector
	if args.sites_save:
		print("Creating boolean vector of sites surviving filters.")
		siteVec = np.zeros(M_old, dtype=np.uint8)
		if args.filter_sites is not None:
			filtVec = M_f.astype(bool)
		else:
			filtVec = np.ones(M_old, dtype=np.uint8).astype(bool)
		if args.maf > 0.0:
			if args.hwe is not None:
				siteArr = np.arange(M_old, dtype=int)[filtVec][maskMAF][maskHWE]
				siteVec[siteArr] = 1
			else:
				siteArr = np.arange(M_old, dtype=int)[filtVec][maskMAF]
				siteVec[siteArr] = 1
		elif args.hwe is not None:
			siteArr = np.arange(M_old, dtype=int)[filtVec][maskHWE]
			siteVec[siteArr] = 1
		else:
			print("All sites have been kept.")
			siteVec[:] = 1
		np.savetxt(f"{args.out}.sites", siteVec, fmt="%i")
		print("Saved boolean vector of sites kept after filtering as " + \
			f"{args.out}.sites")
		del siteVec, filtVec
	
	# Print elapsed time for estimation
	t_tot = time()-start
	t_min = int(t_tot//60)
	t_sec = int(t_tot - t_min*60)
	print(f"Total elapsed time: {t_min}m{t_sec}s")
	
	# Write output info to log-file
	with open(args.out + ".log", "a") as log:
		if converged:
			log.write(f"\nPCAngsd converged in {it} iterations using {K} " + \
				"eigenvectors.\n")
		else:
			log.write(f"\nPCAngsd did not converge using {K} eigenvectors!\n")
		log.write(f"Saved covariance matrix as {args.out}.cov\n")
		if args.selection:
			log.write(f"Saved test statistics as {args.out}.selection\n")
		if args.snp_weights:
			log.write(f"Saved SNP weights as {args.out}.weights\n")
		if args.pcadapt:
			log.write(f"Saved z-scores as {args.out}.pcadapt.zscores\n")
		if args.inbreed_sites:
			log.write("Saved per-site inbreeding coefficients as " + \
				f"{args.out}.inbreed.sites\n")
			log.write(f"Saved likelihood ratio tests as {args.out}.lrt.sites\n")
		if args.inbreed_samples:
			log.write("Saved per-sample inbreeding coefficients as " + \
				f"{args.out}.inbreed.samples\n")
		if args.post:
			log.write(f"Saved called genotype posteriors as {args.out}.post\n")
		if args.post_inbreed:
			log.write(f"Saved called genotype posteriors as {args.out}.post.inbreed\n")
		if args.geno is not None:
			log.write("Saved called genotype matrix as " + \
				f"{args.out}.geno.npy (Binary).\n")
		if args.geno_inbreed is not None:
			log.write("Saved called genotype matrix as " + \
				f"{args.out}.geno.inbreed.npy (Binary).\n")
		if args.tree:
			log.write(f"Saved newick tree as {args.out}.tree\n")
			log.write(f"Saved tree covariance matrix as {args.out}.tree.cov\n")
		if args.admix:
			if args.admix_auto:
				log.write("Estimated admixture using alpha-search. " + \
					f"Alpha={aB},\tLog-likelihood={lB}\n")
			log.write(f"Saved admixture proportions as {args.out}.admix.{a_K}.Q\n")
			log.write("Saved ancestral allele frequencies proportions as " + \
				f"{args.out}.admix.{a_K}.P\n")
		if args.maf_save:
			log.write(f"Saved minor allele frequencies as {args.out}.freqs\n")
		if args.dosage_save:
			log.write(f"Saved genotype dosages as {args.out}.dosage.npy (Binary)\n")
		if args.pi_save:
			log.write("Saved individual allele frequencies as " + \
				f"{args.out}.pi.npy (Binary)\n")
		if args.sites_save:
			log.write("Saved boolean vector of sites kept after filtering as " + \
				f"{args.out}.sites")
		log.write(f"\nTotal elapsed time: {t_min}m{t_sec}s\n")

		

##### Define main #####
if __name__ == "__main__":
	main()
