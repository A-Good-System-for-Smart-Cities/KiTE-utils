from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import calibration_curve
from KiTE.metrics import ELCE2
from sklearn.metrics import pairwise_distances
import matplotlib.pylab as plt
import numpy as np
import os
import matplotlib as mpl



def example_1():


    train_samples = 20000  # Samples used for training the models
    cv_samples = 20000  # Samples used for training the models

    X_train = X[:train_samples]
    X_cv = X[train_samples : train_samples + cv_samples]
    X_test = X[train_samples + cv_samples :]

    y_train = y[:train_samples]
    y_cv = y[train_samples : train_samples + cv_samples]
    y_test = y[train_samples + cv_samples :]

    # kernel hyperparameter
    gamma = 1.0 / np.median(pairwise_distances(X_test, metric="euclidean")) ** 2

    # Create classifiers
    lr = LogisticRegression()
    gnb = GaussianNB()
    svc = LinearSVC(C=1.0)
    rfc = RandomForestClassifier()
    print("Done making classifiers")
    """
    for clf, name in [(lr, 'Logistic'),
                      (gnb, 'Naive Bayes'),
                      (svc, 'Support Vector Classification'),
                      (rfc, 'Random Forest')]:
        clf.fit(X_train, y_train)
        if hasattr(clf, "predict_proba"):
            prob_pos = clf.predict_proba(X_test)[:, 1]
        else:  # use decision function
            prob_pos = clf.decision_function(X_test)
            prob_pos = \
                (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())

        X_loc = np.zeros(X_test.shape[0])[:, np.newaxis]
        test_estimate, null, p_value =  \
            MLCE2_test_estimator(X_loc, y_test, prob_pos, prob_kernel_wdith=0.2, iterations=100,
                                 kernel_function='rbf', gamma = 1.0, verbose=False)
        print(name, test_estimate, p_value)
    """
    #############################################################################
    # Plot calibration plots

    plt.figure(figsize=(16, 9))
    ax1 = plt.subplot2grid((7, 2), (0, 0), rowspan=4)
    ax2 = plt.subplot2grid((7, 2), (4, 0), rowspan=2)
    ax3 = plt.subplot2grid((7, 2), (6, 0))

    ax4 = plt.subplot2grid((7, 2), (0, 1), rowspan=4)
    ax5 = plt.subplot2grid((7, 2), (4, 1), rowspan=2)
    ax6 = plt.subplot2grid((7, 2), (6, 1))

    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    ax4.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    ax1.hist(X_test, y_test, bins=20, histtype='step')
    # ax1.hist(X_test, p, b)

    # for clf, name in [
    #     (lr, "Logistic"),
    #     (gnb, "Naive Bayes"),
    #     (svc, "Support Vector Classification"),
    #     (rfc, "Random Forest"),
    # ]:
    #     clf.fit(X_train, y_train)
    #     if hasattr(clf, "predict_proba"):
    #         prob_pos = clf.predict_proba(X_test)[:, 1]
    #         prob_cv = clf.predict_proba(X_cv)[:, 1]
    #     else:  # use decision function
    #         prob_pos = clf.decision_function(X_test)
    #         prob_pos = (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())
    #
    #         prob_cv = clf.decision_function(X_cv)
    #         prob_cv = (prob_cv - prob_cv.min()) / (prob_cv.max() - prob_cv.min())
    #
    #     # prob_cal = calibrate(X_cv, y_cv, prob_cv, prob_test=prob_pos, method="isotonic")
    #     # fraction_of_positives, mean_predicted_value = calibration_curve(
    #     #     y_test, prob_pos, n_bins=20
    #     # )
    #     #
    #     # ax1.plot(
    #     #     mean_predicted_value, fraction_of_positives, "s-", label="%s" % (name,)
    #     # )
    #     #
    #     # ax2.hist(prob_pos, range=(0, 1), bins=20, label=name, histtype="step", lw=2)
    #     #
    #     # fraction_of_positives, mean_predicted_value = calibration_curve(
    #     #     y_test, prob_cal, n_bins=20
    #     # )
    #     # ax4.plot(
    #     #     mean_predicted_value,
    #     #     fraction_of_positives,
    #     #     "o-",
    #     #     label="%s (calibrated)" % (name,),
    #     # )
    #
    #     ax5.hist(prob_cal, range=(0, 1), bins=20, label=name, histtype="step", lw=2)
    #
    #     ELCE2_ = ELCE2(
    #         X_test,
    #         y_test,
    #         prob_pos,
    #         prob_kernel_wdith=0.1,
    #         kernel_function="rbf",
    #         gamma=gamma,
    #     )
    #     ax3.plot(ELCE2 * 100, 1, "v", markersize=14, markeredgewidth=2)
    #
    #     ELCE2_ = ELCE2(
    #         X_test,
    #         y_test,
    #         prob_cal,
    #         prob_kernel_wdith=0.1,
    #         kernel_function="rbf",
    #         gamma=gamma,
    #     )
    #     ax6.plot(ELCE2_, 1, "v", markersize=14, markeredgewidth=2)
    #
    # ax1.set_ylabel("Fraction of positives")
    # ax1.set_ylim([0.0, 1.0])
    # ax1.set_xlim([0.0, 1.0])
    # ax1.legend(loc="lower right")
    # ax1.set_title("Calibration plots (reliability curve) -- uncalibrated models")
    #
    # ax2.set_xlabel("Mean predicted value")
    # ax2.set_ylabel("PDF")
    # ax2.legend(loc="upper center", ncol=2)
    # ax2.set_yticks([])
    #
    # ax3.set_xlabel(r"ELCE$^2_{u}$", size=18)
    # ax3.set_yticks([])
    # # ax3.set_xlim([0.0, 50.])
    #
    # ax4.set_ylabel("Fraction of positives")
    # ax4.set_ylim([0.0, 1.0])
    # ax4.set_xlim([0.0, 1.0])
    # ax4.legend(loc="lower right")
    # ax4.set_title("Calibration plots (reliability curve) -- calibrated models")
    #
    # ax5.set_xlabel("Mean predicted value")
    # ax5.set_ylabel("PDF")
    # ax5.legend(loc="upper center", ncol=2)
    # ax5.set_yticks([])
    #
    # ax6.set_xlabel(r"ELCE$^2_{u}$", size=20)
    # ax6.set_yticks([])
    # ax6.set_xlim([0.0, 50.])
    plt.savefig("ex1.png")
example_1()
