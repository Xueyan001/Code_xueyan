import numpy as np

def my_variance(xs: np.ndarray) -> np.ndarray:
    """ Computes the sample variance of a given vector of scalars.

    Args:
        xs: 1D numpy array containing scalars

    Returns:
        The empirical variance of the provided vector as a float
    """
    n = len(xs)
    mean = np.mean(xs)
    variance = np.sum((xs - mean) ** 2) / (n - 1)
    return variance

#print(my_variance(np.array([1, 1, 1])))

def my_mse(z1: np.ndarray, z2: np.ndarray) -> np.ndarray:
    """ Computes the Mean Squared Error (MSE)

    Args:
        z1: A 1D numpy array (usually the predictions).
        z2: Another 1D numpy array.

    Returns:
        The MSE of the given data.
    """
    mse = np.mean((z1 - z2) ** 2)
    return mse

#print(my_mse(np.array([3.0]), np.array([4.0])))