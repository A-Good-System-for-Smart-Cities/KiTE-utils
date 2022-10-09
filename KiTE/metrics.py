"""
Metrics utilities to help test if a model is locally biased.
"""

from joblib import Parallel, delayed
from sklearn.gaussian_process.kernels import pairwise_kernels
import logging
import numpy as np
from KiTE.validation import check_attributes


def _calculate_err_vector(Y, p):
    return Y - p


def ELCE2_estimator(K_xx, err):
    """
    The estimator ELCE^2

    Parameters
    ----------
    K_xx : numpy-array
        evaluated kernel function

    err : numpy-array
        one-dimensional error vector

    Returns
    -------
    numpy-array
        estimated ELCE^2
    """
    K = (err.flatten() * K_xx.T).T * err.flatten()
    return K.sum() - K.diagonal().sum()


def ELCE2_normalization(K):
    """
    The normalization of estimator ELCE^2

    Parameters
    ----------
    K : numpy-array
        evaluated kernel function

    Returns
    -------
    float
        estimated normalization of ELCE^2
    """

    size = K.shape[0]

    return (size - 1.0) * K.sum() / size


def ELCE2_null_estimator(err, K, rng):
    """
    Compute the ELCE^2_u for one bootstrap realization

    Parameters
    ----------
    err : numpy-array
        one-dimensional error vector

    K : numpy-array
        evaluated kernel function

    rng : type(np.random.RandomState())
        numpy random function

    Returns
    -------
    float
        unbiased estimate of ELCE^2_u

    """

    # randomize error vector so error = sample .. not looking at local neighbors
    # checking local calibration ..
    # randomizaiton -- quanitifies noise in estimator
    idx = rng.permutation(len(err))

    return ELCE2_estimator(K, err[idx])


def compute_null_distribution(
    p_err, K, iterations=1000, n_jobs=1, verbose=False, random_state=None
):
    """
    Compute the null-distribution of test statistics via a bootstrap procedure

    Parameters
    ----------
    p_err : numpy-array
        one-dimensional probability error vector

    K : numpy-array
        evaluated kernel function

    iterations : int
        controls the number of bootstrap realizations

    verbose : bool
        controls the verbosity of the model's output

    random_state : type(np.random.RandomState()) or None
        defines the initial random state

    Returns
    -------
    numpy-array
        boostrap samples of the test null distribution
    """
    rng = (
        random_state
        if isinstance(random_state, type(np.random.RandomState()))
        else np.random.RandomState(random_state)
    )
    iterations_list = range(iterations)

    # compute the null distribution
    # for 1 cpu run the normal code, for more cpu use the Parallel library. This maximize the speed.
    test_null = (
        [ELCE2_null_estimator(p_err, K, rng) for _ in iterations_list]
        if n_jobs == 1
        else Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(ELCE2_null_estimator)(p_err, K, rng) for _ in iterations_list
        )
    )

    return np.array(test_null)


def ELCE2(
    X,
    Y,
    p,
    kernel_function="rbf",
    prob_kernel_width=0.1,
    iterations=None,
    use_diffusion_distance=False,
    verbose=True,
    random_state=None,
    n_jobs=1,
    **kwargs,
):
    """
    This function estimate ELCE^2_u employing a kernel trick. ELCE^2_u tests if a proposed posterior credible interval
    is calibrated employing a randomly drawn calibration test. The null hypothesis is that the posteriors are
    properly calibrated. This function perform a bootstrap algorithm to estimate the null distribution,
    and corresponding p-value.

    Parameters
    ----------
    X : numpy-array
        data, of size NxD [N is the number of data points, D is the features dimension]

    Y : numpy-array
        credible error vector, of size Nx1 [N is the number of data points]

    p : numpy-array
        probability vector, of size Nx1 [N is the number of data points]

    kernel_function : string
        defines the kernel function. For the list of implemented kernels, please consult with [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.kernel_metrics.html#sklearn.metrics.pairwise.kernel_metrics)

    prob_kernel_width : float
        Width of the probably kernel function

    iterations : int
        controls the number of bootstrap realizations

    use_diffusion_distance : bool
        Use diffusion instead of eucliden distances
        Transforms X into a diffusion space (see diffusion_maps.py)

    verbose : bool
        controls the verbosity of the model's output

    random_state : type(np.random.RandomState()) or None
        defines the initial random state

    n_jobs : int
        number of jobs to run in parallel

    **kwargs : **kwargs
        extra parameters, these are passed to `pairwise_kernels()` as kernel parameters
        E.g., if `kernel_two_sample_test(..., kernel_function='rbf', gamma=0.1)`

    Returns
    -------
    tuple
        - SIZE = 1 if iterations=`None` else 3 (float, numpy-array, float)
            - first element is the test value,
            - second element is samples from the null distribution via a bootstraps algorithm,
            - third element is the estimated p-value.
    """

    def create_kernel():
        """
        RBF = https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.rbf_kernel.html?highlight=gaussian+kernel

        Returns
        -------
        A kernel matrix K such that K_{i, j} is the kernel between the ith and jth vectors of the given matrix X, if Y is None.
        """
        # Pre-compute Kernel Function (Hyperplane/Convolution)
        K_pp_gamma = 1.0 / (prob_kernel_width**2)
        K_pp_metric = "rbf"

        # In binary class (p vs 1-p) vs miltiple classification (p1 + ...+ pn = 1)
        # p should be nx1 for the kernal function
        if len(p.shape) == 2:
            K_pp_data = p
        elif len(p.shape) == 1:  # if p.shape == 1 ... turn n, --> nx1
            K_pp_data = p[:, np.newaxis]
        else:
            raise ValueError(
                f"p has invalid dimensions of {p.shape}. The length of p's shape should be 1 or 2, not {len(p.shape)}"
            )

        K_xx = pairwise_kernels(X, X, metric=kernel_function, **kwargs)
        K_pp = pairwise_kernels(
            K_pp_data, K_pp_data, metric=K_pp_metric, gamma=K_pp_gamma
        )

        # Consider both similar features && similar probabilities
        # K_xx includes ENTIRE feature space ... may not match input as model ..
        K = K_pp * K_xx
        return K

    check_attributes(X, Y)
    K = create_kernel()
    assert len(Y) == len(p)
    assert None not in Y
    assert None not in p
    p_err = _calculate_err_vector(Y, p)

    # Estimate the Null "Oracle" Distribution
    test_value = ELCE2_estimator(K, p_err)
    norm = ELCE2_normalization(K)

    if verbose:
        logging.info(f"test value = {(test_value / norm)}")
        logging.info("Computing the null distribution.")

    if iterations is None:
        return test_value / norm

    # p-value's resolution
    resolution = 1.0 / iterations

    # compute the null distribution via a bootstrap algorithm
    test_null = compute_null_distribution(
        p_err,
        K,
        iterations=iterations,
        verbose=verbose,
        n_jobs=n_jobs,
        random_state=random_state,
    )

    # Permutation -- est noise under Ho ... Ho assume no global/local miscallibrate .. should cent around 0
    # center it at zero (to account for global mis-calibration)
    test_null -= np.mean(test_null)

    # compute the p-value, if less then the resolution set it to the resolution
    p_value = max(resolution, resolution * (test_null > test_value).sum())

    if verbose:
        msg = (
            (f"p-value < {p_value} \t (resolution : {resolution})")
            if p_value == resolution
            else (f"p-value ~= {p_value} \t (resolution : {resolution})")
        )
        logging.info(msg)

    return test_value / norm, test_null / norm, p_value
