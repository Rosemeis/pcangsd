# PCAngsd

**Version 0.97** 
*PCAngsd now supports PLINK files!*

Framework for analyzing low depth next-generation sequencing (NGS) data in heterogeneous populations using principal component analysis (PCA). Population structure is inferred to detect the number of significant principal components which is used to estimate individual allele frequencies using genotype dosages in a SVD model. The estimated individual allele frequencies are then used in an probabilistic framework to update the genotype dosages such that an updated set of individual allele frequencies can be estimated iteratively based on inferred population structure. A covariance matrix can be estimated using the updated prior information of the estimated individual allele frequencies.

The estimated individual allele frequencies and principal components can be used as prior knowledge in other probabilistic methods based on a same Bayesian principle. PCAngsd can perform the following analyses: 

* Covariance matrix
* Genotype calling
* Admixture
* Inbreeding coefficients (both per-individual and per-site)
* HWE test
* PC-based genome-wide selection scan
* Kinship matrix

The entire framework is written Python 2.7 based on Numpy data structures to take use of the Numba library for improving performances in bottlenecks. Multithreading has been added to take advantage of multiple cores and is highly recommended.

## Get PCAngsd
```
git clone https://github.com/Rosemeis/pcangsd.git
cd pcangsd/
```

### Install dependencies
The required set of Python packages are easily installed using the pip command and the python_packages.txt file included in the pcangsd folder.
```
pip install --user -r python_packages.txt
```

## Usage
A full wiki of how to use all the features of PCAngsd is available at [popgen.dk](http://www.popgen.dk/software/index.php/PCAngsd). 

PCAngsd is used by running the main caller file pcangsd.py. To see all available options use the following command:
```
python pcangsd.py -h
```

The only input PCAngsd needs is estimated genotype likelihoods in Beagle format. These can be estimated using [ANGSD](https://github.com/ANGSD/angsd).
New functionality for using PLINK files has been added (version 0.9). Genotypes are automatically converted into a genotype likelihood matrix.


## Citation
Please cite our paper in GENETICS:
[Inferring Population Structure and Admixture Proportions in Low-Depth NGS Data](http://www.genetics.org/content/210/2/719).
