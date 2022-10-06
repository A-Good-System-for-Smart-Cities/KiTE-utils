from scipy.spatial.distance import pdist, squareform
import numpy as np
from numpy.linalg import matrix_power, eig # BRUTE FORCE
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import euclidean_distances
from scipy.linalg import eigh

# ASSERT num_lambdas < min(num_rows, num_cols)


def calculate_kernel_matrix(X, epsilon):
    """
    Calculate Kernel Matrix ... an indication of local geometry
    NOTE: K is symmetric (k(x,y)=k(y,x)) and positivity preserving (k(x,y) >= 0 forall x,y)

    Parameters
    ----------
    K : numpy-array
        kernel matrix

    Returns
    -------
    numpy-array
        Full Connectivity Matrix
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
    dx = K.sum(axis = 0)
    assert(0 not in dx)

    # 2. p_ij = k_ij / d_row
    # How handle floating point issues???
    p = np.multiply(1/dx, K)
    return p

def transform_into_diffusion_space(K=None, num_timesteps = 1, num_eigenvectors = 10):
    p = get_connectivity_matrix(K)

    # Eigendecomposition:
    eigenvalues, eigenvectors = eigh(p)#np.linalg.eig(p) [n-3, n-1]

    # Build Diffy Map:
    n = len(eigenvalues)
    assert(len(eigenvalues) == len(eigenvectors))
    diffy_map = []
    for i in range(num_eigenvectors):
        indx_from_back = n - i - 1
        val = (eigenvalues[n - i - 1]) ** num_timesteps
        vec = eigenvectors[n - i - 1]
        diffy_map.append(val*vec)

    return diffy_map

def calculate_diffusion_distance_matrix(diffy_map):
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
    diffy_distance = pairwise_distances(diffy_map, metric='euclidean')
    return diffy_distance
