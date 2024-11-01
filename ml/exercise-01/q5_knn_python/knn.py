import numpy as np


def knn(samples, k):
    # compute density estimation from samples with KNN
    # Input
    #  samples    : DxN matrix of data points
    #  k          : number of neighbors
    # Output
    #  estDensity : estimated density in the range of [-5, 5]

    #####Insert your code here for subtask 5b#####
    # Compute the number of the samples created

    # Compute the number of samples created
    N = len(samples)  # compute the length of sample
    pos = np.arange(-5, 5.0, 0.1)  # Returns a 100 dimensional vector,np is numpy lib

    # K-Nearest Neighbor(K-NN) density estimation = K / NV, fix K, determine V
    # V = 2 * distance to the k-th nearest neighbor
    distance = np.sort(np.abs(pos[np.newaxis, :] - samples[:, np.newaxis]), axis=0)
    pde = (k / (2 * N)) / distance[k - 1, :]

    # output
    estDensity = np.stack((pos, pde), axis=1)


    return estDensity
