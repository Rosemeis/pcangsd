# PCAngsd

**Version 0.98**
*I have rewritten PCAngsd to be based on Cython for speed and parallelization and is now compatible with any newer Python version. The older version based on the Numba library (only working with Python 2.7) is still available in version 0.973.*

Framework for analyzing low depth next-generation sequencing (NGS) data in heterogeneous populations using principal component analysis (PCA). Population structure is inferred to detect the number of significant principal components which is used to estimate individual allele frequencies using genotype dosages in a SVD model. The estimated individual allele frequencies are then used in an probabilistic framework to update the genotype dosages such that an updated set of individual allele frequencies can be estimated iteratively based on inferred population structure. A covariance matrix can be estimated using the updated prior information of the estimated individual allele frequencies.

The estimated individual allele frequencies and principal components can be used as prior knowledge in other probabilistic methods based on a same Bayesian principle. PCAngsd can perform the following analyses: 

* Covariance matrix
* Genotype calling
* Admixture
* Inbreeding coefficients (both per-individual and per-site)
* HWE test
* Genome selection scan
* Kinship matrix


## Get PCAngsd and build
```
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
```
python pcangsd.py -h
```

Since version 0.98, PCAngsd is now outputting solely in binary Numpy format (.npy). In order to read files in python:
```python
import numpy as np
C = np.load("output.cov.npy") # Reads in estimated covariance matrix 
```

R can also read Numpy matrices using the "RcppCNPy" library:
```R
library(RcppCNPy)
C <- npyLoad("output.cov.npy") # Reads in estimated covariance matrix
```


The only input PCAngsd needs is estimated genotype likelihoods in Beagle format. These can be estimated using [ANGSD](https://github.com/ANGSD/angsd).
New functionality for using PLINK files has been added (version 0.9). Genotypes are automatically converted into a genotype likelihood matrix where the user can incorporate an error model.


## Citation
Please cite our papers:

Population structure: [Inferring Population Structure and Admixture Proportions in Low-Depth NGS Data](http://www.genetics.org/content/210/2/719).\
HWE test: [Testing for Hardy‐Weinberg Equilibrium in Structured Populations using Genotype or Low‐Depth NGS Data](https://onlinelibrary.wiley.com/doi/abs/10.1111/1755-0998.13019).