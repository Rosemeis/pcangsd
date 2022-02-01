# PCAngsd

**Version 1.10**

Framework for analyzing low-depth next-generation sequencing (NGS) data in heterogeneous/structured populations using principal component analysis (PCA). Population structure is inferred by estimating individual allele frequencies in an iterative approach using a truncated SVD model. The covariance matrix is estimated using the estimated individual allele frequencies as prior information for the unobserved genotypes in low-depth NGS data.

The estimated individual allele frequencies can further be used to account for population structure in other probabilistic methods. PCAngsd can perform the following analyses:

* Covariance matrix
* Admixture estimations
* Inbreeding coefficients (both per-individual and per-site)
* HWE test
* Genome-wide selection scans
* Genotype calling
* Estimate NJ tree of samples


## Get PCAngsd and build
### Dependencies
The PCAngsd software relies on the following Python packages that you can install through conda (recommended) or pip:

- numpy
- cython
- scipy

You can create an environment through conda easily or install dependencies through pip as follows:
```
# Conda environment
conda env create -f environment.yml

# pip
pip install --user -r requirements.txt
```

## Install and build
```bash
git clone https://github.com/Rosemeis/pcangsd.git
cd pcangsd
python setup.py build_ext --inplace
pip3 install -e .
```

You can now run PCAngsd with the `pcangsd` command.

## Usage
### Running PCAngsd
PCAngsd works directly on genotype likelihood files or PLINK files.
```bash
# See all options
pcangsd -h

# Genotype likelihood file in Beagle format with 2 eigenvectors using 64 threads
pcangsd -b input.beagle.gz -e 2 -t 64 -o output.pcangsd

# PLINK files (using file-prefix, *.bed, *.bim, *.fam)
pcangsd -p input.plink -e 2 -t 64 -o output.pcangsd

# Perform PC-based selection scan and estimate admixture proportions
pcangsd -b input.beagle.gz -e 2 -t 64 -o output.pcangsd --selection --admix
```

PCAngsd will output files in text or binary Numpy format (.npy). In order to read files in python:
```python
import numpy as np
C = np.genfromtxt("output.pcangsd.cov") # Reads in estimated covariance matrix (text)
D = np.load("output.pcangsd.selection.npy") # Reads PC based selection statistics
```

R can also read Numpy matrices using the "RcppCNPy" R library:
```R
library(RcppCNPy)
C <- as.matrix(read.table("output.pcangsd.cov")) # Reads in estimated covariance matrix
D <- npyLoad("output.pcangsd.selection.npy") # Reads PC based selection statistics

# Plot PCA plot
e <- eigen(C)
plot(e$vectors[,1:2], xlab="PC1", ylab="PC2", main="PCAngsd")

# Obtain p-values from selection scan
p <- pchisq(D, 1, lower.tail=FALSE)
```

Beagle genotype likelihood files can be generated from BAM files using [ANGSD](https://github.com/ANGSD/angsd). For inference of population structure in genotype data with non-random missigness, we recommend our [EMU](https://github.com/Rosemeis/emu) software that performs accelerated EM-PCA, however with fewer functionalities than PCAngsd (#soon).


## Citation
Please cite our papers:

Population structure: [Inferring Population Structure and Admixture Proportions in Low-Depth NGS Data](http://www.genetics.org/content/210/2/719).\
HWE test: [Testing for Hardy‐Weinberg Equilibrium in Structured Populations using Genotype or Low‐Depth NGS Data](https://onlinelibrary.wiley.com/doi/abs/10.1111/1755-0998.13019).\
Selection: [Detecting Selection in Low-Coverage High-Throughput Sequencing Data using Principal Component Analysis](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-021-04375-2).
