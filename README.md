# PCAngsd

**Version 0.2**

Framework for analyzing low depth next-generation sequencing data in heterogeneous populations using principal component analysis (PCA). Population structure is inferred by PCA and the principal components are used to estimate individual allele frequencies using a linear regression model. The individual allele frequencies are then used in an Empirical Bayes approach to estimate the posterior genotype probabilities in order to estimate a new covariance matrix and its corresponding principal components. PCAngsd performs this iterative update until the individual allele frequencies have converged. 

The estimated individual allele frequencies and principal components can be used as prior knowledge in other probabilistic methods based on an Empirical Bayes approach. PCAngsd can perform the following analyses: 

* Covariance matrix
* Genotype calling
* Inbreeding coefficients (both per-individual and per-site)
* Genome selection scan
* Kinship matrix

The entire framework is written Python 2.7 using two external popular python packages; Numpy for fast optimized numerical computations and Pandas for easy data frame manipulations.

## Get PCAngsd
```
git clone https://github.com/Rosemeis/pcangsd.git
cd pcangsd/
```

## Usage
A full wiki of how to use all the features of PCAngsd is available at [popgen.dk](http://www.popgen.dk/software/index.php/PCAngsd). 

PCAngsd is used by running the main caller file pcangsd.py. To see all available options use the following command:
```python
python pcangsd.py -h
```

The only input PCAngsd needs and accepts is the estimated genotype likelihoods in BEAGLE format. These can be estimated using [ANGSD](https://github.com/ANGSD/angsd).
