import numpy as np
cimport numpy as np
from cython.parallel import prange
from cython import boundscheck, wraparound
from libc.math cimport log

##### Cython functions for inbreed.py #####
### Per-site
@boundscheck(False)
@wraparound(False)
cpdef inbreedSites_update(float[:,::1] L, float[:,::1] P, float[::1] F, int t):
    cdef int m = P.shape[0]
    cdef int n = P.shape[1]
    cdef int i, s
    cdef float expH, prob0, prob1, prob2, prior0, prior1, prior2, priorSum, \
                temp0, temp1, temp2, tempSum, Fadj
    with nogil:
        for s in prange(m, num_threads=t):
            expH = 0.0
            prob0 = 0.0
            prob1 = 0.0
            prob2 = 0.0
            for i in range(n):
                Fadj = (1 - P[s,i])*P[s,i]*F[s]
                prior0 = max(1e-4, (1 - P[s,i])*(1 - P[s,i]) + Fadj)
                prior1 = max(1e-4, 2*P[s,i]*(1 - P[s,i]) - 2*Fadj)
                prior2 = max(1e-4, P[s,i]*P[s,i] + Fadj)
                priorSum = prior0 + prior1 + prior2

                # Readjust distribution
                prior0 = prior0/priorSum
                prior1 = prior1/priorSum
                prior2 = prior2/priorSum

                # Posterior
                temp0 = L[s,2*i+0]*prior0
                temp1 = L[s,2*i+1]*prior1
                temp2 = (1.0 - L[s,2*i+0] - L[s,2*i+1])*prior2
                tempSum = temp0 + temp1 + temp2

                # Sum over individuals
                prob0 = prob0 + temp0/tempSum
                prob1 = prob1 + temp1/tempSum
                prob2 = prob2 + temp2/tempSum

                # Count heterozygotes
                expH = expH + 2*P[s,i]*(1 - P[s,i])

            # ANGSD procedure
            prob0 = max(1e-4, prob0/<float>(n))
            prob1 = max(1e-4, prob1/<float>(n))
            prob2 = max(1e-4, prob2/<float>(n))

            # Update the inbreeding coefficient
            F[s] = 1 - (n*prob1/expH)
            F[s] = min(max(-1.0, F[s]), 1.0)

### Log-likelihoods
@boundscheck(False)
@wraparound(False)
cpdef loglike(float[:,::1] L, float[:,::1] P, float[::1] F, float[::1] T, int t):
    cdef int m = P.shape[0]
    cdef int n = P.shape[1]
    cdef int i, s
    cdef float prior0, prior1, prior2, priorSum, like0, like1, like2, Fadj, \
                logAlt, logNull
    with nogil:
        for s in prange(m, num_threads=t):
            logAlt = 0.0
            logNull = 0.0
            for i in range(n):
                ### Alternative model
                Fadj = (1 - P[s,i])*P[s,i]*F[s]
                prior0 = max(1e-4, (1 - P[s,i])*(1 - P[s,i]) + Fadj)
                prior1 = max(1e-4, 2*P[s,i]*(1 - P[s,i]) - 2*Fadj)
                prior2 = max(1e-4, P[s,i]*P[s,i] + Fadj)
                priorSum = prior0 + prior1 + prior2

                # Readjust distribution
                prior0 = prior0/priorSum
                prior1 = prior1/priorSum
                prior2 = prior2/priorSum

                # Likelihood*prior
                like0 = L[s,2*i+0]*prior0
                like1 = L[s,2*i+1]*prior1
                like2 = (1.0 - L[s, 2*i+0] - L[s, 2*i+1])*prior2
                logAlt = logAlt + log(like0 + like1 + like2)

                ### Null model
                like0 = L[s,2*i+0]*(1 - P[s,i])*(1 - P[s,i])
                like1 = L[s,2*i+1]*2*P[s,i]*(1 - P[s,i])
                like2 = (1.0 - L[s,2*i+0] - L[s,2*i+1])*P[s,i]*P[s,i]
                logNull = logNull + log(like0 + like1 + like2)
            T[s] = 2*(logAlt - logNull)

### Per-individual
@boundscheck(False)
@wraparound(False)
cpdef inbreedSamples_update(float[:,::1] L, float[:,::1] P, float[::1] F, int t):
    cdef int m = P.shape[0]
    cdef int n = P.shape[1]
    cdef int i, s
    cdef float expH, prob0, prob1, prob2, prior0, prior1, prior2, priorSum, \
                temp0, temp1, temp2, tempSum, Fadj
    with nogil:
        for i in prange(n, num_threads=t):
            expH = 0.0
            prob0 = 0.0
            prob1 = 0.0
            prob2 = 0.0
            for s in range(m):
                Fadj = (1 - P[s,i])*P[s,i]*F[i]
                prior0 = max(1e-4, (1 - P[s,i])*(1 - P[s,i]) + Fadj)
                prior1 = max(1e-4, 2*P[s,i]*(1 - P[s,i]) - 2*Fadj)
                prior2 = max(1e-4, P[s,i]*P[s,i] + Fadj)
                priorSum = prior0 + prior1 + prior2

                # Readjust distribution
                prior0 = prior0/priorSum
                prior1 = prior1/priorSum
                prior2 = prior2/priorSum

                # Posterior
                temp0 = L[s,2*i+0]*prior0
                temp1 = L[s,2*i+1]*prior1
                temp2 = (1.0 - L[s,2*i+0] - L[s,2*i+1])*prior2
                tempSum = temp0 + temp1 + temp2

                # Sum over individuals
                prob0 = prob0 + temp0/tempSum
                prob1 = prob1 + temp1/tempSum
                prob2 = prob2 + temp2/tempSum

                # Count heterozygotes
                expH = expH + 2*P[s,i]*(1 - P[s,i])

            # ANGSD procedure
            prob0 = max(1e-4, prob0/<float>(m))
            prob1 = max(1e-4, prob1/<float>(m))
            prob2 = max(1e-4, prob2/<float>(m))

            # Update the inbreeding coefficient
            F[i] = 1 - (m*prob1/expH)
            F[i] = min(max(-1.0, F[i]), 1.0)
