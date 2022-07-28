import numpy as np
from sklearn.gaussian_process.kernels import pairwise_kernels
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
from netcal.scaling import TemperatureScaling, BetaCalibration
from netcal.binning import HistogramBinning, BBQ, ENIR
from KiTE_utils.validation import check_attributes
from KiTE_utils.calibration_models import KRR_calibration, EWF_calibration


# TODO -- use Case??
def local_bias_estimator(X, Y, p, X_grid, model="KRR", kernel_function="rbf", **kwargs):
    """
    Estimates Local Bias

    Parameters
    ----------
        X: numpy-array
            feature data
        Y: numpy-array
            label data
        p: numpy-array
            same dimensions as y
        model: str
            options = ['KRR', 'SVR', 'EWF']
            Type of model to train and compare results in bias calculation
        kernel_function: str
            See sklearn documentation for pairwise_kernels for valid kernel_functions
        **kwargs:
            Additional arguments required for the selected kernel_function

    Returns
    ----------
        bias: numpy-array
            Matrix with bias calculation
    """
    # Checks Matrix Compatibility
    check_attributes(X, Y)

    # Trains a kernelized model
    options = ["KRR", "SVR", "EWF"]
    if model == "KRR":
        model = KernelRidge(kernel=kernel_function, **kwargs)
    elif model == "SVR":
        model = SVR(kernel=kernel_function, **kwargs)
    elif model == "EWF":
        K = pairwise_kernels(X, X_grid, metric=kernel_function, **kwargs)
        p_err = Y - p
        bias = np.sum(p_err.flatten() * K.T, axis=1) / np.sum(K.T, axis=1)
        return bias
    else:
        raise ValueError(
            f"Model {model} is not defined. Supported models include: {options}"
        )

    bias_calibration = Y - p

    model.fit(X, bias_calibration)
    bias = model.predict(X_grid)

    return bias


def construct_credible_error_vector(Y, Yr_up, Yr_down, alpha):
    """
    For a one dimensional output prediction Y it construct the credible error vector. It takes the lower and upper percentiles and assuming credible level alpha is fixed.

    Parameters
    ----------
        Y: numpy-array
            data, of size Nx1 [N is the number of data points]

        Yr_up: numpy-array
            upper percentile vector, of size Nx1 [N is the number of data points]

        Yr_up: numpy-array
            upper percentile vector, of size Nx1 [N is the number of data points]

        alpha: float
            the theoretical credible level alpha

    return
    ------
    numpy-array: credible error vector
    """

    if Y.flatten().shape[0] != Yr_up.flatten().shape[0]:
        raise ValueError(
            "Incompatible dimension for Y and Yr_up matrices. Y and Yr_up should have the same feature dimension,"
            f": Y.shape[0] == {Y.shape[0]} while Yr.shape[0] == {Yr_up.shape[0]}."
        )

    if Y.flatten().shape[0] != Yr_down.flatten().shape[0]:
        raise ValueError(
            "Incompatible dimension for Y and Yr matrices. Y and Yr should have the same feature dimension,"
            f": Y.shape[0] == {Y.shape[0]} while Yr_down.shape[0] == {Yr_down.shape[0]}"
        )

    if alpha < 0 or alpha > 1:
        raise ValueError(
            f"Incompatible value for alpha. alpha should be a real value between 0 and 1: alpha == {alpha}"
        )

    # indicator of Y less than posterior percentile r
    Yind = 1.0 * ((Y < Yr_up) * (Y > Yr_down))

    # percentile/probability error vector
    e = Yind - alpha

    return e


def calibrate(
    Xtrain, prob_train, Ytrain, Xtest=None, prob_test=None, method="platt", **kwargs
):
    """
    A calibration method that takes the predicted probabilties and positive cases and recalibrates the probabilities.

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
    p_calibrated : array, shape (n_bins,)
        The calibrated error for test set.


    References
    ----------
    Küppers et al., "Multivariate Confidence Calibration for Object Detection." CVPR Workshops, 2020.

    Leeuw, Hornik, Mair, Isotone, "Optimization in R : Pool-Adjacent-Violators Algorithm (PAVA) and Active
    Set Methods." Journal of Statistical Software, 2009.

    Naeini, Mahdi Pakdaman, Gregory Cooper, and Milos Hauskrecht, "Obtaining well calibrated probabilities
    using bayesian binning." Twenty-Ninth AAAI Conference on Artificial Intelligence, 2015.

    Kull, Meelis, Telmo Silva Filho, and Peter Flach: "Beta calibration: a well-founded and easily implemented
    improvement on logistic calibration for binary classifiers." Artificial Intelligence and Statistics,
    PMLR 54:623-631, 2017.

    Zadrozny, Bianca and Elkan, Charles: "Obtaining calibrated probability estimates from decision
    trees and naive bayesian classifiers." In ICML, pp. 609–616, 2001.

    Zadrozny, Bianca and Elkan, Charles: "Transforming classifier scores into accurate
    multiclass probability estimates." In KDD, pp. 694–699, 2002.

    Ryan J Tibshirani, Holger Hoefling, and Robert Tibshirani: "Nearly-isotonic regression."
    Technometrics, 53(1):54–61, 2011.

    Naeini, Mahdi Pakdaman, and Gregory F. Cooper: "Binary classifier calibration using an ensemble of near
    isotonic regression models." 2016 IEEE 16th International Conference on Data Mining (ICDM). IEEE, 2016.

    Chuan Guo, Geoff Pleiss, Yu Sun and Kilian Q. Weinberger: "On Calibration of Modern Neural Networks."
    Proceedings of the 34th International Conference on Machine Learning, 2017.

    Pereyra, G., Tucker, G., Chorowski, J., Kaiser, L. and Hinton, G.: “Regularizing neural networks by
    penalizing confident output distributions.” CoRR, 2017.

    Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P.,
    Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M. and Duchesnay, E.:
    "Scikit-learn: Machine Learning in Python." In Journal of Machine Learning Research, volume 12 pp 2825-2830,
    2011.

    Platt, John: "Probabilistic outputs for support vector machines and comparisons to regularized likelihood
    methods." Advances in large margin classifiers, 10(3): 61–74, 1999.

    Neumann, Lukas, Andrew Zisserman, and Andrea Vedaldi: "Relaxed Softmax: Efficient Confidence Auto-Calibration
    for Safe Pedestrian Detection." Conference on Neural Information Processing Systems (NIPS) Workshop MLITS, 2018.

    Nilotpal Chakravarti, Isotonic Median Regression: A Linear Programming Approach, Mathematics of Operations
    Research Vol. 14, No. 2 (May, 1989), pp. 303-308.
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
        raise ValueError(f"Method {method} is not defined.")

    p_calibrated[np.isnan(p_calibrated)] = 0

    # normalize the large numbers and small numbers to one and zero
    # p_calibrated[p_calibrated > 1.0] = 1.0
    # p_calibrated[p_calibrated < 0.0] = 0.0

    return p_calibrated


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
    score : float
        calibration error score

    References
    ----------
    Alexandru Niculescu-Mizil and Rich Caruana (2005) Predicting Good
    Probabilities With Supervised Learning, in Proceedings of the 22nd
    International Conference on Machine Learning (ICML).
    """
    # compute fraction of positive cases per y_prob in a bin. See scikit-learn documentation for details
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
        raise ValueError(f"Method {method} is not defined.")
