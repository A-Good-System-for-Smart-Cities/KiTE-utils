from joblib import Parallel, delayed
from sklearn.gaussian_process.kernels import pairwise_kernels
from tqdm import tqdm
from KiTE import no_none_arg
from KiTE.validation import check_attributes
import logging
import numpy as np


@no_none_arg
def ELCE2_estimator(K_xx, err):
    """
    The estimator $ELCE^2 = \sum (e Kxx e^T) / n / (n-1)$

    Parameters
    ----------
        err: numpy array
            one-dimensional error vector. ... make X, into just X ...

        K_xx: numpy array
            evaluated kernel function.

    return
    ------
        float: estimated ELCE^2
    """
    K = (err.flatten() * K_xx.T).T * err.flatten()
    return K.sum() - K.diagonal().sum()


@no_none_arg
def ELCE2_normalization(K):
    """
    The normalization of estimator ELCE^2 = \sum (1 x Kxx x 1T) / n / (n-1)

    Parameters
    ----------
        K: numpy array
            evaluated kernel function.

    return
    ------
        float: estimated normalization of ELCE^2
    """

    size = K.shape[0]

    return (size - 1.0) * K.sum() / size


@no_none_arg
def ELCE2_null_estimator(err, K, rng):
    """
    Compute the ELCE^2_u for one bootstrap realization.

    Parameters
    ----------
        err: numpy-array
            one-dimensional error vector.

        K: numpy-array
            evaluated kernel function.

        rng: type(np.random.RandomState())
             a numpy random function

    return
    ------
        float: an unbiased estimate of ELCE^2_u
    """

    idx = rng.permutation(len(err))

    return ELCE2_estimator(K, err[idx]) #randomize error vector .. err = sample .. not looking at local neighbords
    # cehcking local calibration .. each guy should be at 0 ...
    # randomizaiton -- quanitfyes noise in estimator


@no_none_arg
def _calculate_err_vector(Y, p):
    return Y - p


def compute_null_distribution(
    p_err, K, iterations=1000, n_jobs=1, verbose=False, random_state=None
):
    """
    Compute the null-distribution of test statistics via a bootstrap procedure.

    Parameters
    ----------
        p_err: numpy-array
            one-dimensional probability error vector.

        K: numpy array
            evaluated kernel function.

        iterations: int
            controls the number of bootstrap realizations

        verbose: bool
            controls the verbosity of the model's output.

        random_state: type(np.random.RandomState()) or None
            defines the initial random state.

    return
    ------
    numpy-array: a boostrap samples of the test null distribution
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
    verbose=False,
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
        X: numpy-array
            data, of size NxD [N is the number of data points, D is the features dimension]

        Y: numpy-array
            credible error vector, of size Nx1 [N is the number of data points]

        p: numpy-array
            probability vector, of size Nx1 [N is the number of data points]

        kernel_function: string
            defines the kernel function. For the list of implemented kernel please consult with
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.kernel_metrics.html#sklearn.metrics.pairwise.kernel_metrics

        prob_kernel_width: float
            Width of the probably kernel function.

        iterations: int
            controls the number of bootstrap realizations

        verbose: bool
            controls the verbosity of the model's output.

        random_state: type(np.random.RandomState()) or None
            defines the initial random state.

        n_jobs: int
            number of jobs to run in parallel.

        **kwargs:
            extra parameters, these are passed to `pairwise_kernels()` as kernel parameters o
            as the number of k. E.g., if `kernel_two_sample_test(..., kernel_function='rbf', gamma=0.1)`

    return
    ----------
    tuple of size 1 -- if iterations=`None` -- or 3 (float, numpy-array, float)
        - first element is the test value,
        - second element is samples from the null distribution via a bootstraps algorithm,
        - third element is the estimated p-value.
    """

    def create_kernel():
        """
        RBF = https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.rbf_kernel.html?highlight=gaussian+kernel
        Returns: A kernel matrix K such that K_{i, j} is the kernel between the ith and jth vectors of the given matrix X, if Y is None.
        """
        # Pre-compute Kernel Function (Hyperplane/Convolution)
        K_pp_gamma = 1.0 / prob_kernel_width ** 2
        K_pp_metric = 'rbf' #radial basis functin = RBF ... gamma = 1/(2L^2) ... based on 1/2sigma^2... #kernel_function  #'rbf' #Should this be hardcoded as rbf?

        # In binary class (p vs 1-p) vs miltiple classification (p1 + ...+ pn = 1)
        # Rn, only works for 2d classfier
        # p should be nx1 for the kernal function
        K_pp_data = ( # if p.shape == 1 ... turn n, --> nx1
            p if len(p.shape) == 2 else p[:, np.newaxis] if len(p.shape) == 1 else None
        )
        if not K_pp_data:
            raise ValueError(
                f"p has invalid dimensions of {p.shape}. The length of p's shape should be 1 or 2, not {len(p.shape)}"
            )

        # Assume valid p and X
        # Safe default = gaussian kernal
        # Kxx -- examine feature space while Kpp = neighbors in probability space
        # Kxx = 1 -- Global calibration .. focus only on Kpp = output probability
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
    # center it at zero (to account for global mis-calibration) <-- WHY?? .. Are we allowing for bias when should assume no bias?
    test_null -= np.mean(test_null) # What would mbe noise of a Ho model?

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