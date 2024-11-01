import numpy as np
from getLogLikelihood import getLogLikelihood


def MStep(gamma, X):
    # Maximization step of the EM Algorithm
    #
    # INPUT:
    # gamma          : NxK matrix of responsibilities for N datapoints and K Gaussians.
    # X              : Input data (NxD matrix for N datapoints of dimension D).
    #
    # N is number of data points
    # D is the dimension of the data points
    # K is number of Gaussians
    #
    # OUTPUT:
    # logLikelihood  : Log-likelihood (a scalar).
    # means          : Mean for each gaussian (KxD).
    # weights        : Vector of weights of each gaussian (1xK).
    # covariances    : Covariance matrices for each component(DxDxK).

    #####Insert your code here for subtask 6c#####

    N, D = X.shape  # Number of data points and dimension of data points
    K = gamma.shape[1]  # Number of Gaussians

    # Initialize parameters
    means = np.zeros((K, D))
    covariances = np.zeros((D, D, K))

    # Calculate the effective number of points assigned to each cluster
    Nk = np.sum(gamma, axis=0)

    # Update means
    for k in range(K):
        # Calculate the weighted sum of points for cluster k
        weighted_sum = np.dot(gamma[:, k], X)
        # Update mean for cluster k
        means[k] = weighted_sum / Nk[k]

    # Update weights
    weights = Nk / N

    # Update covariances
    for k in range(K):
        # Initialize a covariance matrix for cluster k
        covariance_matrix = np.zeros((D, D))
        for n in range(N):
            # Calculate the outer product of the mean differences
            diff = (X[n] - means[k]).reshape(D, 1)
            covariance_matrix += gamma[n, k] * np.dot(diff, diff.T)
        # Normalize the covariance matrix by the effective number of points
        covariances[:, :, k] = covariance_matrix / Nk[k]

    # Compute the log likelihood using the updated parameters
    logLikelihood = getLogLikelihood(means, weights, covariances, X)


    return weights, means, covariances, logLikelihood
