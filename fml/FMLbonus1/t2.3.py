import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import kmeans_plusplus

def my_kmeans(xs, init_centers, n_iter):
    centers = init_centers
    for _ in range(n_iter):
        # Assign points to the nearest center
        labels = np.argmin(np.linalg.norm(xs[:, np.newaxis] - centers, axis=2), axis=1)

        # Recalculate centers
        new_centers = np.array([xs[labels == k].mean(axis=0) for k in range(len(centers))])

        # Check for convergence (optional)
        if np.all(centers == new_centers):
            break
        centers = new_centers

    return centers


def generate_toy_dataset():
    np.random.seed(0)  # For reproducibility
    points = []
    means = [(-2, 2), (-2, -2), (2, -2), (2, 2)]
    covariances = [np.array([[0.2, 0], [0, 0.2]]), np.array([[0.2, 0], [0, 0.2]]),
                   np.array([[0.5, 0], [0, 0.5]]), np.array([[0.5, 0], [0, 0.5]])]
    probabilities = [0.3, 0.2, 0.4, 0.1]

    for _ in range(100):
        # Choose a distribution and generate a point
        index = np.random.choice(len(means), p=probabilities)
        point = np.random.multivariate_normal(means[index], covariances[index])
        points.append(point)

    points = np.array(points)
    plt.scatter(points[:, 0], points[:, 1])
    plt.title('Toy Dataset Scatter Plot')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()

    return points

def my_plot(points):
    """ Plots the K-Means result for different numbers of cluster given 2-dimensional data.

    Notes:
        Use the `kmeans_plusplus` function to get initial cluster centers.

    Args:
        xs: A 2D numpy array of shape (N, 2) containing N 2-dimensional samples.
    """

    plt.figure(figsize=(10, 10))
    n_clusters = [2, 3, 4, 5]  # different numbers of clusters

    # iterate over each cluster n in `n_clusters` with index i
    for i, n in enumerate(n_clusters):
        plt.subplot(2, 2, i + 1)
        # YOUR CODE HERE
        init_centers = points[np.random.choice(points.shape[0], n, replace=False)]
        # Run our implementation of K-Means
        final_centers = my_kmeans(points, init_centers, n_iter=5)
        plt.scatter(points[:, 0], points[:, 1], label='Data Points')
        plt.scatter(final_centers[:, 0], final_centers[:, 1], color='red', label='Centers')
        plt.title(f'K-Means with K={n}')
        plt.legend()
        plt.tight_layout()

    plt.show()
    #raise NotImplementedError()




points = generate_toy_dataset()

my_plot(points)