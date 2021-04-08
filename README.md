# PCAngsd

**Version 1.0**
*I have reworked a lot of the parallelization in PCAngsd and removed some of its clunky prototype features. Python 3.x versions will only be targeted in future updates (however, it might still be compatible with v.2.7).*

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
```bash
git clone https://github.com/Rosemeis/pcangsd.git
cd pcangsd/
python setup.py build_ext --inplace
```

### Install dependencies
The required set of Python packages are easily installed using the pip command and the requirements.txt file included in the pcangsd folder.
```
pip install --user -r requirements.txt
```

## Usage
A full wiki of how to use all the features of PCAngsd is available at [popgen.dk](http://www.popgen.dk/software/index.php/PCAngsd).

PCAngsd is used by running the main caller file pcangsd.py. To see all available options use the following command:
```bash
python pcangsd.py -h

# Genotype likelihoods using 64 threads
python pcangsd.py -beagle input.beagle.gz -out output -threads 64

# PLINK files (using file-prefix, *.bed, *.bim, *.fam)
python pcangsd.py -beagle input.plink -out output -threads 64
```

PCAngsd will mostly output files in binary Numpy format (.npy) with a few exceptions. In order to read files in python:
```python
import numpy as np
C = np.genfromtxt("output.cov") # Reads in estimated covariance matrix (text)
D = np.load("output.selection.npy") # Reads PC based selection statistics
```

R can also read Numpy matrices using the "RcppCNPy" R library:
```R
library(RcppCNPy)
C <- as.matrix(read.table("output.cov")) # Reads in estimated covariance matrix
D <- npyLoad("output.selection.npy") # Reads PC based selection statistics
```


PCAngsd accepts either genotype likelihoods in Beagle format or PLINK genotype files. Beagle files can be generated from BAM files using [ANGSD](https://github.com/ANGSD/angsd). For inference of population structure in genotype data with non-random missigness, we recommend our [EMU](https://github.com/Rosemeis/emu) software that performs accelerated EM-PCA, however with fewer functionalities than PCAngsd (#soon).


## Citation
Please cite our papers:

Population structure: [Inferring Population Structure and Admixture Proportions in Low-Depth NGS Data](http://www.genetics.org/content/210/2/719).\
HWE test: [Testing for Hardy‐Weinberg Equilibrium in Structured Populations using Genotype or Low‐Depth NGS Data](https://onlinelibrary.wiley.com/doi/abs/10.1111/1755-0998.13019).
Selection: [Detecting Selection in Low-Coverage High-Throughput Sequencing Data using Principal Component Analysis](https://www.biorxiv.org/content/10.1101/2021.03.01.432540v1.abstract).
