import numpy as np
def getLogLikelihood(means, weights, covariances, X):
    # Log Likelihood estimation
    #
    # INPUT:
    # means          : Mean for each Gaussian KxD
    # weights        : Weight vector 1xK for K Gaussians
    # covariances    : Covariance matrices for each gaussian DxDxK
    # X              : Input data NxD
    # where N is number of data points
    # D is the dimension of the data points
    # K is number of gaussians
    #
    # OUTPUT:
    # logLikelihood  : log-likelihood

    #####Insert your code here for subtask 6a#####
    if len(X.shape) > 1:
        N, D = X.shape ## D is the dimension of the data points,means the feature of one point
    else:
        N = 1
        D = X.shape[0]

    K = len(weights) # get number of gaussians distribution
    logLikelihood = 0
    for i in range(N):  # For each of the data points
        p = 0 # initialization probability p
        for j in range(K):  # For each of the mixture components

            if N == 1:
                meansDiff = X - means[j]
            else:
                meansDiff = X[i,:] - means[j]

            covariance = covariances[:, :, j].copy() # copy the j-th covariance matrix
            # norm = 1 / sqrt((2pi^D) * |covariance matrix|)
            norm = 1. / float(((2 * np.pi) ** (float(D) / 2.)) * np.sqrt(np.linalg.det(covariance)))

            # Ax=b, x = (A^(-1))b
            p += weights[j] * norm * np.exp(-0.5 * ((meansDiff.T).dot(np.linalg.lstsq(covariance.T, meansDiff.T)[0].T)))
        logLikelihood += np.log(p)


    return logLikelihood

