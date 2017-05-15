# pcangsd

**Version 0.1**

Population genetic analyses performed using principal components for next-generation sequencing data. Using the principal components estimated from the covariance matrix to make a linear model of the genotype dosages to estimate the individual allele frequencies of each individual in all sites. The individual allele frequencies can then be used in an iterative manner due to a bayesian approach in estimating posterior genotypes to produce a new covariance matrix and its corresponding principal components, which once again can be used to estimate a set of updated individual allele frequencies. PCAngsd performs this iterative update until convergence.

The estimated individual allele frequencies and principal components can then be used as prior knowledge in various population genetic analyses where some are already implemented in PCAngsd (Covariance matrix, Inbreeding, Selection scan, Kinship).

The entire program is written Python 2.7 using two external popular python packages; Numpy for fast numerical computations and Pandas for easy data frame manipulations.

## Get PCAngsd
```
git clone https://github.com/Rosemeis/pcangsd.git
cd pcangsd/
```

## Usage
PCAngsd is used by running the main caller file pcangsd.py. To see all available options use the following command:
```
python pcangsd.py -h
```

The only input PCAngsd needs and accepts is the estimated genotype likelihoods in BEAGLE format. These can be estimated using [ANGSD](https://github.com/ANGSD/angsd).
