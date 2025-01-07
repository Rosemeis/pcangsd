####################
# Run pcadapt scan #
####################
#
#	args[1]: zscores matrix
#	args[2]: prefix for output files
#

args = commandArgs(trailingOnly=TRUE)
library(bigutilsr)

zscores <- read.table(args[1])[,1]
K <- ncol(zscores)

# For one component only
if (K == 1) {
	d2 <- (zscores - median(zscores))^2
} else {
	d2 <- dist_ogk(zscores)
}

write.table(d2, file=paste0(args[2], ".pcadapt.test.txt"), quote=F, row.names=F, col.names=F)
write.table(pchisq(d2, df=K, lower.tail=F), file=paste0(args[2], ".pcadapt.pval.txt"), quote=F, row.names=F, col.names=F)
