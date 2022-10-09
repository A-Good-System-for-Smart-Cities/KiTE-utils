"""
Utilities to transform coordinates and distances into a diffusion space.
"""

import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import euclidean_distances
from scipy.linalg import eigh


def calculate_kernel_matrix(X, epsilon):
    """
    Calculate Kernel Matrix ... an indication of local geometry
    NOTE: K is symmetric (k(x,y)=k(y,x)) and positivity preserving (k(x,y) >= 0 forall x,y)

    Parameters
    ----------
    X : numpy-array
        Input data

    epsilon : float
        Metric for kernel width

    Returns
    -------
    numpy-array
        Kernel Matrix
    """
    distance_sq = euclidean_distances(X, X, squared=True)
    K = np.exp(-distance_sq / (2.0 * epsilon))

    return K


def get_connectivity_matrix(K):
    """
    Calculate connectivity matrix (as normalization of Kernel Matrix Rows).
    Each value in connectivity matrix is probability of stepping to another cell in 1 timestep!
    Multiply connectivity matrices to see how probabilities change over 2 timesteps

    NOTES:
        - p is positivity preserving (p(x,y) >= 0 forall x,y) & sum of P in each row = 1


    P_i = collection of all connectivities leading to point xi
        = Diffusion Space of X

    Parameters
    ----------
    K : numpy-array
        kernel matrix

    Returns
    -------
    numpy-array
        Full Connectivity Matrix
    """

    # 1. d_row = sum across K's 1st dimension
    dx = K.sum(axis=0)
    assert 0 not in dx

    # 2. p_ij = k_ij / d_row
    p = np.multiply(1 / dx, K)
    return p


def transform_into_diffusion_space(K=None, num_timesteps=1, num_eigenvectors=10):
    """
    Given Kernel, calculates connectivity matrix (as normalization of Kernel Matrix Rows).
    Performs Eigendecomposition to transform kernel coordinates into a diffusion space

    Parameters
    ----------
    K : numpy-array
        kernel matrix
    num_timesteps : int
        Number of timesteps for diffusion calculation
    num_eigenvectors : int
        Number of eigenvectors (used in descending order of magnitude) used to calculate diffusion coordinates

    Returns
    -------
    numpy-array
        Diffusion Coordinates/Map
    """
    p = get_connectivity_matrix(K)

    # Eigendecomposition:
    eigenvalues, eigenvectors = eigh(p)

    # Build Diffy Map:
    n = len(eigenvalues)
    assert len(eigenvalues) == len(eigenvectors)
    diffy_map = []
    for i in range(num_eigenvectors):
        indx_from_back = n - i - 1
        val = (eigenvalues[indx_from_back]) ** num_timesteps
        vec = eigenvectors[indx_from_back]
        diffy_map.append(val * vec)

    return diffy_map


def calculate_diffusion_distance_matrix(diffy_map):
    """
    New Distance based on pairwise distance between 2 points' connectivity
    D^2 = sum_u [ (P^t)_iu - (P^t)_ju ]^2 ... D = dist(p_iu, p_ij)

    D(xi, xj) ^ 2 = (Pi - Pj)^2

    Parameters
    ----------
    diffy_map : numpy-array
        Diffusion Coordinates/Map

    Returns
    -------
    numpy-array
        Diffusion Distance Matrix
    """
    diffy_distance = pairwise_distances(diffy_map, metric="euclidean")
    return diffy_distance
