import numpy as np
from sklearn.gaussian_process.kernels import pairwise_kernels
from sklearn.kernel_ridge import KernelRidge
from KiTE_utils.validation import check_attributes


class KRR_calibration:
    def __init__(self):
        self.model = "KRR"

    def create_Xp(self, X, p):
        Xp = np.concatenate((X, p[:, np.newaxis]), axis=1) if self.use_y else X.copy()
        return Xp

    def fit(self, X, p, Y, use_y=True, kernel_function="rbf", **kwargs):
        check_attributes(X, Y)
        self.use_y = use_y
        Xp = self.create_Xp(X, p)
        self.model = KernelRidge(kernel=kernel_function, **kwargs)
        observed_bias = Y - p
        self.model.fit(Xp, observed_bias)
        return self.model

    def predict(self, X, p, mode="prob"):
        Xp = self.create_Xp(X, p)
        if mode == "bias":
            return self.model.predict(Xp)
        if mode == "prob":
            return self.model.predict(Xp) + p.flatten()
        raise ValueError(f"Mode {mode} is not defined.")


class EWF_calibration:
    def __init__(self):
        self.model = "EWF"

    def fit(self, X, p, Y, kernel_function="rbf", **kwargs):
        check_attributes(X, Y)

        self.Xp = np.concatenate((X, p[:, np.newaxis]), axis=1)
        self.bias = Y - p

        self.kernel_function = kernel_function
        self.kwargs = kwargs

    def predict(self, Xtest, ptest, mode="prob"):
        Xtestp = np.concatenate((Xtest, ptest[:, np.newaxis]), axis=1)
        K = pairwise_kernels(
            self.Xp, Xtestp, metric=self.kernel_function, **self.kwargs
        )
        bias = np.sum(self.bias.flatten() * K.T, axis=1) / np.sum(K.T, axis=1)

        if mode == "bias":
            return bias
        if mode == "prob":
            return bias + ptest.flatten()
        raise ValueError(f"Mode {mode} is not defined.")
