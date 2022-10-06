"""
Calibration utilities to help reduce the local bias of a given model.
"""

import numpy as np
from sklearn.gaussian_process.kernels import pairwise_kernels
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from netcal.scaling import TemperatureScaling, BetaCalibration
from netcal.binning import HistogramBinning, BBQ, ENIR
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
from KiTE.calibration_models import KRR_calibration, EWF_calibration
from KiTE.validation import check_attributes, check_credible_vector_args


def local_bias_estimator(X, Y, p, X_grid, model="KRR", kernel_function="rbf", **kwargs):
    """
    Estimates model bias by calculating bias = Y - model predictions. Assumes model is already close to the oracle model.

    Parameters
    ----------
    X : numpy-array
        data, of size NxD [N is the number of data points, D is the features dimension]

    Y : numpy-array
        credible error vector, of size Nx1 [N is the number of data points]

    p : numpy-array
        probability vector, of size Nx1 [N is the number of data points]

    X_grid : numpy-array ... QUESTION
        - For EWF, X_grid used to build a kernel. This kernel used to calculate bias
        - Else, X_grid used as "test features" to predict bias

    model : str
        Indicates model type. Valid Options include:
            - "KRR": KernelRidge
            - "SVR": SVR
            - "EWF": EWF

    kernel_function : str
        Indicates kernel function type.
        Refer to sklearn documentation to see which kernel functions are valid for model chosen.

    **kwargs : **kwargs
        extra parameters passed to `pairwise_kernels()` as kernel parameters

    Returns
    -------
    numpy-array
        estimated bias
    """

    check_attributes(X, Y)

    if (
        model == "KRR"
    ):  # look at locals and fit linear model ... capiture .. preocmputed = weights
        model = KernelRidge(kernel=kernel_function, **kwargs)
    elif model == "SVR":
        model = SVR(kernel=kernel_function, **kwargs)
    elif model == "EWF":
        # X = diffy distance .. X_grid = D_new guys (in calibrate set) // X ~ D (OG training set )
        K = pairwise_kernels(X, X_grid, metric=kernel_function, **kwargs)
        p_err = Y - p
        bias = np.sum(p_err.flatten() * K.T, axis=1) / np.sum(K.T, axis=1)
        return bias
    else:
        raise ValueError(
            f"Model {model} is not defined. Valid Options include: KRR, SVR, or EWF"
        )

    # Use chosen model to predict bias_calibration based on features X.
    bias_calibration = Y - p
    model.fit(X, bias_calibration)
    bias = model.predict(X_grid)
    return bias


def construct_credible_error_vector(Y, Yr_up, Yr_down, alpha):
    """
    For a one dimensional output prediction Y, construct the credible error vector.
    Uses given lower and upper percentiles. Assumes credible level alpha is fixed.

    Parameters
    ----------
    Y : numpy-array
        data, of size Nx1 [N is the number of data points]

    Yr_up : numpy-array
        upper percentile vector, of size Nx1 [N is the number of data points]

    Yr_down : numpy-array
        lower percentile vector, of size Nx1 [N is the number of data points]

    alpha : float
        the theoretical credible level alpha

    Returns
    -------
    numpy-array
        Credible Error Vector
    """
    check_credible_vector_args(Y, Yr_up, Yr_down, alpha)

    # indicator of Y less than posterior percentile r
    Yind = 1.0 * ((Y < Yr_up) * (Y > Yr_down))

    # percentile/probability error vector
    e = Yind - alpha

    return e


def calibrate(
    Xtrain, prob_train, Ytrain, Xtest=None, prob_test=None, method="platt", **kwargs
):
    """
    A calibration method that takes the predicted probabilties and positive cases and recalibrate the probabilities.

    Parameters
    ----------
    y_true : array, shape (n_samples_train,)
        True targets for the training set.

    y_prob_train : array, shape (n_samples_train,)
        Probabilities of the positive class to train a calibration model.

    y_prob_test : array, shape (n_samples_test,)
        Probabilities of the positive class to be calibrated (test set). If None it re-calibrate the training set.

    method: string, 'platt', 'isotonic', 'temperature_scaling', 'beta', 'HB', 'BBG', 'ENIR'
        The method to use for calibration. Can be ‘sigmoid’ which corresponds to Platt’s method
        (i.e. a logistic regression model) or ‘isotonic’ which is a non-parametric approach.
        It is not advised to use isotonic calibration with too few calibration samples (<<1000) since it tends to overfit.

    Returns
    -------
    array, shape (n_bins,)
        The calibrated error for test set. (p_calibrated)

    """
    probs = prob_train[:, np.newaxis] if prob_test is None else prob_test[:, np.newaxis]
    Xtest = Xtrain if Xtest is None else Xtest

    if method == "platt":
        model = LogisticRegression()
        model.fit(prob_train[:, np.newaxis], Ytrain)  # LR needs X to be 2-dimensional
        p_calibrated = model.predict_proba(probs)[:, 1]

    elif method == "isotonic":
        model = IsotonicRegression(out_of_bounds="clip")
        model.fit(prob_train, Ytrain)  # LR needs X to be 2-dimensional
        p_calibrated = model.transform(probs.flatten())

    elif method == "temperature_scaling":
        model = TemperatureScaling()
        model.fit(prob_train, Ytrain)
        p_calibrated = model.transform(probs)

    elif method == "beta":
        model = BetaCalibration()
        model.fit(prob_train, Ytrain)
        p_calibrated = model.transform(probs)

    elif method == "HB":
        model = HistogramBinning()
        model.fit(prob_train, Ytrain)
        p_calibrated = model.transform(probs)

    elif method == "BBQ":
        model = BBQ()
        model.fit(prob_train, Ytrain)
        p_calibrated = model.transform(probs)

    elif method == "ENIR":
        model = ENIR()
        model.fit(prob_train, Ytrain)
        p_calibrated = model.transform(probs)

    elif method == "KRR":
        model = KRR_calibration()
        model.fit(Xtrain, prob_train, Ytrain, **kwargs)
        p_calibrated = model.predict(Xtest, probs.flatten(), mode="prob")

    elif method == "EWF":
        model = EWF_calibration()
        model.fit(Xtrain, prob_train, Ytrain, **kwargs)
        p_calibrated = model.predict(Xtest, probs.flatten(), mode="prob")

    else:
        raise ValueError("Method %s is not defined." % method)

    p_calibrated[np.isnan(p_calibrated)] = 0

    # normalize the large numbers and small numbers to one and zero
    # p_calibrated[p_calibrated > 1.0] = 1.0
    # p_calibrated[p_calibrated < 0.0] = 0.0

    return p_calibrated  # f-hat -- f(x) = oracle = f-hat + b(x)


def _counts_per_bin(prob, n_bins):
    """
    Taken from https://github.com/scikit-learn/scikit-learn/blob/a24c8b46/sklearn/calibration.py#L513
    """

    bins = np.linspace(0.0, 1.0 + 1e-8, n_bins + 1)
    binids = np.digitize(prob, bins) - 1
    bin_total = np.bincount(binids, minlength=len(bins))

    nonzero = bin_total != 0
    return bin_total[nonzero]


def calibration_error(y_true, y_prob, n_bins=10, method="ECE"):
    """
    Compute calibration error given true targets and predicted probabilities.
     Calibration curves may also be referred to as reliability diagrams.

    Parameters
    ----------
    y_true : array, shape (n_samples,)
        True targets.

    y_prob : array, shape (n_samples,)
        Probabilities of the positive class.

    method : string, default='ECE', {'ECE', 'MCE', 'BS'}
        Which method to be used to compute calibration error.

    n_bins : int
        Number of bins. Note that a bigger number requires more data.

    Returns
    -------
    float
        calibration error score
    """

    # compute fraction of positive cases per y_prob in a bin.
    # See scikit-learn documentation for details
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_prob, normalize=False, n_bins=n_bins, strategy="uniform"
    )

    if method == "ECE":
        hist_count = _counts_per_bin(y_prob, n_bins)
        return np.sum(
            hist_count * np.abs(fraction_of_positives - mean_predicted_value)
        ) / np.sum(hist_count)
    elif method == "MCE":
        return np.max(np.abs(fraction_of_positives - mean_predicted_value))
    elif method == "BS":
        return brier_score_loss(y_true, y_prob, pos_label=1)
    else:
        raise ValueError("Method %s is not defined." % method)
