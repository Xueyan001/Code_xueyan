import numpy as np
from getLogLikelihood import getLogLikelihood


def EStep(means, covariances, weights, X):
    # Expectation step of the EM Algorithm
    #
    # INPUT:
    # means          : Mean for each Gaussian KxD
    # weights        : Weight vector 1xK for K Gaussians
    # covariances    : Covariance matrices for each Gaussian DxDxK
    # X              : Input data NxD
    #
    # N is number of data points
    # D is the dimension of the data points
    # K is number of Gaussians
    #
    # OUTPUT:
    # logLikelihood  : Log-likelihood (a scalar).
    # gamma          : NxK matrix of responsibilities for N datapoints and K Gaussians.

    #####Insert your code here for subtask 6b#####

    logLikelihood = getLogLikelihood(means, weights, covariances, X)
    N, D = X.shape
    K = len(weights) # get number of gaussians distribution

    gamma = np.zeros((N, K)) # initialization responsibility” of component j for x.
    for i in range(N):  # For each of the data points
        weighted_probs = np.zeros(K)
        for j in range(K):  # For each of the mixture components
            meansDiff = X[i, :] - means[j]
            covariance = covariances[:, :, j].copy()  # copy the j-th covariance matrix
            # norm = 1 / sqrt((2pi^D) * |covariance matrix|)
            norm = 1. / float(np.sqrt((2 * np.pi) ** (float(D))) * np.sqrt(np.linalg.det(covariance)))
            weighted_probs[j] += weights[j] * norm * np.exp(-0.5 * ((meansDiff.T).dot(np.linalg.lstsq(covariance.T, meansDiff.T)[0].T)))
            gamma[i, :] = weighted_probs / np.sum(weighted_probs)


    return [logLikelihood, gamma]
