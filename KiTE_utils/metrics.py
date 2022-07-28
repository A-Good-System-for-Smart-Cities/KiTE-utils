import numpy as np
from joblib import Parallel, delayed
from sklearn.gaussian_process.kernels import pairwise_kernels
from tqdm import tqdm
from KiTE_utils.validation import check_attributes


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

    return ELCE2_estimator(K, err[idx])


def ELCE2_estimator(K_xx, err):
    """
    The estimator $ELCE^2 = \sum (e Kxx e^T) / n / (n-1)$

    Parameters
    ----------
        err: numpy array
            one-dimensional error vector.

        K_xx: numpy array
            evaluated kernel function.

    return
    ------
        float: estimated ELCE^2
    """

    # size = len(err)

    K = (err.flatten() * K_xx.T).T * err.flatten()

    return K.sum() - K.diagonal().sum()  # / (size * (size-1.0))


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

    return (size - 1.0) * K.sum() / size  # - K.diagonal().sum()


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

    if isinstance(random_state, type(np.random.RandomState())):
        rng = random_state
    else:
        rng = np.random.RandomState(random_state)

    if verbose:
        iterations_list = tqdm(range(iterations))
    else:
        iterations_list = range(iterations)

    # compute the null distribution
    # for 1 cpu run the normal code, for more cpu use the Parallel library. This maximize the speed.
    if n_jobs == 1:
        test_null = [ELCE2_null_estimator(p_err, K, rng) for _ in iterations_list]

    else:
        test_null = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(ELCE2_null_estimator)(p_err, K, rng) for _ in iterations_list
        )
    return np.array(test_null)


def ELCE2(
    X,
    Y,
    p,
    kernel_function="rbf",
    prob_kernel_wdith=0.1,
    iterations=None,
    verbose=False,
    random_state=None,
    n_jobs=1,
    **kwargs,
):
    """
    This function estimate ELCE^2_u employing a kernel trick. ELCE^2_u tests if a proposed posterior credible interval
      is calibrated employing a randomly drawn calibration test. The null hypothesis is that the posteriors are
      properly calibrated This function perform a bootstrap algorithm to estimate the null distribution,
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

        prob_kernel_wdith: float
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

    check_attributes(X, Y, iterations=1000, n_jobs=1)

    # pre-compute the kernel function
    K_xx = pairwise_kernels(X, X, metric=kernel_function, **kwargs)

    if len(p.shape) == 2:
        K_pp = pairwise_kernels(p, p, metric="rbf", gamma=1.0 / prob_kernel_wdith ** 2)
    elif len(p.shape) == 1:
        K_pp = pairwise_kernels(
            p[:, np.newaxis],
            p[:, np.newaxis],
            metric="rbf",
            gamma=1.0 / prob_kernel_wdith ** 2,
        )

    K = K_pp * K_xx

    # error vector
    p_err = Y - p

    # estimate the test value
    test_value = ELCE2_estimator(K, p_err)
    norm = ELCE2_normalization(K)

    if verbose:
        print(f"test value = {(test_value / norm)}")
        print("Computing the null distribution.")

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

    # center it at zero (to account for global mis-calibration)
    test_null -= np.mean(test_null)

    # compute the p-value, if less then the resolution set it to the resolution
    p_value = max(resolution, resolution * (test_null > test_value).sum())

    if verbose:
        if p_value == resolution:
            print(f"p-value < {p_value} \t (resolution : {resolution})")
        else:
            print(f"p-value ~= {p_value} \t (resolution : {resolution})")

    return test_value / norm, test_null / norm, p_value


def error_witness_function(
    X, Y, p, X_grid, p_grid, prob_kernel_wdith=0.1, kernel_function="rbf", **kwargs
):
    """
    This function compute the Error Witness Function (EWF) for a new set of data points defined with `X_grid`,
    `p_grid`.

    Parameters
    ----------
        X: numpy-array
            calibration data, of size NxD [N is the number of data points, D is the features dimension]

        Y: numpy-array
            credible error vector, of size Nx1 [N is the number of calibration data points]

        p: numpy-array
            probability vector, of size Nx1 [N is the number of calibration data points]

        X: numpy-array
            test data, of size MxD [M is the number of test data points, D is the features dimension]

        p: numpy-array
            probability vector, of size Mx1 [M is the number of test data points]

    return
    ----------
    numpy-array.
    """

    check_attributes(X, Y)

    # def witness_function(e, K_xx):
    #    return np.sum(e * K_xx.T, axis=1) / len(e)

    # pre-compute the kernel function
    # K_xx = pairwise_kernels(X, grid, metric=kernel_function, **kwargs)

    # pre-compute the kernel function
    K_xx = pairwise_kernels(X, X_grid, metric=kernel_function, **kwargs)
    K_pp = pairwise_kernels(
        p[:, np.newaxis],
        p_grid[:, np.newaxis],
        metric="rbf",
        gamma=1.0 / prob_kernel_wdith ** 2,
    )
    K = K_pp * K_xx

    # error vector
    p_err = Y - p

    ewf = np.sum(p_err.flatten() * K.T, axis=1) / np.sum(K.T, axis=1)  # / len(p_err)

    return ewf
