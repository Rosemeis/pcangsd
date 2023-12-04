# PCAngsd

**Version 1.21**

Framework for analyzing low-depth next-generation sequencing (NGS) data in heterogeneous/structured populations using principal component analysis (PCA). Population structure is inferred by estimating individual allele frequencies in an iterative approach using a truncated SVD model. The covariance matrix is estimated using the estimated individual allele frequencies as prior information for the unobserved genotypes in low-depth NGS data.

The estimated individual allele frequencies can further be used to account for population structure in other probabilistic methods. *PCAngsd* can perform the following analyses:

* Covariance matrix
* Admixture estimation
* Inbreeding coefficients (both per-individual and per-site)
* HWE test
* Genome-wide selection scans
* Genotype calling
* Estimate NJ tree of samples


## Get PCAngsd and build
### Dependencies
The *PCAngsd* software relies on the following Python packages that you can install through conda (recommended) or pip:

- numpy
- cython
- scipy

You can create an environment through conda easily or install dependencies through pip as follows:
```
# Conda environment
conda env create -f environment.yml

# pip
pip3 install --user -r requirements.txt
```

## Install and build
```bash
git clone https://github.com/Rosemeis/pcangsd.git
cd pcangsd
pip3 install .
```

You can now run *PCAngsd* with the `pcangsd` command.

## Usage
### Running *PCAngsd*
*PCAngsd* works directly on genotype likelihood files or PLINK files.
```bash
# See all options
pcangsd -h

# Genotype likelihood file in Beagle format with 2 eigenvectors using 64 threads
pcangsd -b input.beagle.gz -e 2 -t 64 -o pcangsd
# Outputs by default log-file (pcangsd.log) and covariance matrix (pcangsd.cov)

# PLINK files (using file-prefix, *.bed, *.bim, *.fam)
pcangsd -p input.plink -e 2 -t 64 -o pcangsd

# Perform PC-based selection scan and estimate admixture proportions
pcangsd -b input.beagle.gz -e 2 -t 64 -o pcangsd --selection --admix
# Outputs the following files:
# log-file (pcangsd.log)
# covariance matrix (pcangsd.cov)
# selection statistics (pcangsd.selection)
# admixture proportions (pcangsd.admix.3.Q)
# ancestral allele frequencies (pcangsd.admix.3.F)
```
*PCAngsd* will output most files in text-format.

Quick example of reading output and creating PCA plot in R:
```R
C <- as.matrix(read.table("pcangsd.cov")) # Reads estimated covariance matrix
D <- as.matrix(read.table("pcangsd.selection")) # Reads PC based selection statistics

# Plot PCA plot
e <- eigen(C)
plot(e$vectors[,1:2], xlab="PC1", ylab="PC2", main="PCAngsd")

# Obtain p-values from PC-based selection scan
p <- pchisq(D, 1, lower.tail=FALSE)
```

Read files in python and create PCA plot using matplotlib:
```python
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2
C = np.loadtxt("pcangsd.cov") # Reads estimated covariance matrix
D = np.loadtxt("pcangsd.selection") # Reads PC based selection statistics

# Plot PCA plot
evals, evecs = np.linalg.eigh(C)
evecs = evecs[:,::-1]
plt.scatter(evecs[:,0], evecs[:,1])
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCAngsd")
plt.show()

# Obtain p-values from PC-based selection scan
p = chi2.sf(D, 1)
```

Beagle genotype likelihood files can be generated from BAM files using [ANGSD](https://github.com/ANGSD/angsd). For inference of population structure in genotype data with non-random missigness, we recommend our [EMU](https://github.com/Rosemeis/emu) software that performs accelerated EM-PCA, however with fewer functionalities than *PCAngsd*.


## Citation
Please cite our papers:

Population structure: [Inferring Population Structure and Admixture Proportions in Low-Depth NGS Data](http://www.genetics.org/content/210/2/719).\
HWE test: [Testing for Hardy‐Weinberg Equilibrium in Structured Populations using Genotype or Low‐Depth NGS Data](https://onlinelibrary.wiley.com/doi/abs/10.1111/1755-0998.13019).\
Selection: [Detecting Selection in Low-Coverage High-Throughput Sequencing Data using Principal Component Analysis](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-021-04375-2).
