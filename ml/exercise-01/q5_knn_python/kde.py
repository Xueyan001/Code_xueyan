import numpy as np


def kde(samples, h):
    # compute density estimation from samples with KDE

    # Input
    #  samples    : DxN matrix of data points
    #  h          : (half) window size/radius of kernel
    # Output
    #  estDensity : estimated density in the range of [-5,5]

    #####Insert your code here for subtask 5a#####
    # Compute the number of samples created
    N = len(samples) # compute the length of sample
    pos = np.arange(-5, 5.0, 0.1) # Returns a 100 dimensional vector,np is numpy lib

    # kernel density estimator, 100 * N matrix , sum is N dimension, pde is Probability Density Estimation
    norm = np.sqrt(2 * np.pi * (h**2)) * N
    pde = np.sum(np.exp(-(pos[np.newaxis, :] - samples[:, np.newaxis]) ** 2 / (2 * h ** 2)), axis=0) / norm
    # print("Array dimensions:", pde.shape)

    # output
    estDensity = np.stack((pos, pde), axis=1)


    return estDensity
