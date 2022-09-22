from scipy.spatial.distance import pdist, squareform
import numpy as np
from numpy.linalg import matrix_power # BRUTE FORCE


# ASSERT num_lambdas < min(num_rows, num_cols)
def get_connectivity_matrix(K):
    """
    Calculate connectivity matrix (as normalization of Kernel Matrix Rows).
    Each value in connectivity matrix is probability of stepping to another cell in 1 timestep!
    Multiply connectivity matrices to see how probabilities change over 2 timesteps
    NOTE: Each row's probabilities should sum to 1.

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
    dx = K.sum(axis = 0)
    assert(0 not in dx) #WHAT TO DO IN A SPARSE MATRIX

    # 2. p_ij = k_ij / d_row
    # How handle floating point issues???
    p = np.multiply(1/dx, K)
    return p


def get_timesteped_connectivity_matrix(P, num_timesteps = 1, num_lambdas = 1):
    """
    P = QDQ^T --> Can j do SVD?
        * Diagonalize P and sort eigenvalues and corresponding left eigenvectors in descending order
        * Use truncated set of n eigenvectors to created reduced dimensionality space

    Parameters
    ----------
    P : numpy-array
        Full Connectivity Matrix

    num_timesteps : int > 0

    num_lambdas : int > 0

    Returns
    -------
    numpy-array
        Connectivity Matrix based on truncasted set of num_lambdas eigenvectors

    """

    #Return based on ALL Eigenvalues ... Really costly!
    return matrix_power(P, num_timesteps)


def calculate_diffusion_distance_matrix(P):
    """
    New Distance based on pairwise distance between 2 points' connectivity
    D^2 = sum_u [ (P^t)_iu - (P^t)_ju ]^2 ... D = dist(p_iu, p_ij)

    D(xi, xj) ^ 2 = (Pi - Pj)^2

    Parameters
    ----------
    P : numpy-array
        Connectivity matrix on truncated set of eigenvalues

    Returns
    -------
    numpy-array
        Diffusion Distance Matrix
    """
    diffy_distance = pdist(P, 'euclidean')
    return diffy_distance
