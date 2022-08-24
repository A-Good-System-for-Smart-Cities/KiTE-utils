from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from KiTE.metrics import ELCE2
from KiTE.plots import plot_probability_frequency
from sklearn.metrics import pairwise_distances
import numpy as np
import plotly.figure_factory as ff
import plotly.graph_objs as go
from sklearn.calibration import CalibrationDisplay
import matplotlib.pyplot as plt


def create_train_cv_test_split(X, y, train_samples=20000, cv_samples=20000):
    X_train = X[:train_samples]
    X_cv = X[train_samples : train_samples + cv_samples]
    X_test = X[train_samples + cv_samples :]

    y_train = y[:train_samples]
    y_cv = y[train_samples : train_samples + cv_samples]
    y_test = y[train_samples + cv_samples :]
    return X_train, X_cv, X_test, y_train, y_cv, y_test


def instantiate_classifiers(model="lr", **kwargs):
    return (
        LogisticRegression(**kwargs)
        if model == "lr"
        else GaussianNB(**kwargs)
        if model == "gnb"
        else LinearSVC(**kwargs)
        if model == "svc"
        else RandomForestClassifier(**kwargs)
        if model == "rfc"
        else None
    )


def predict_probability(clf, X_test, X_cv):
    if hasattr(clf, "predict_proba"):
        prob_pos = clf.predict_proba(X_test)[:, 1]
        prob_cv = clf.predict_proba(X_cv)[:, 1]
    else:  # use decision function
        prob_pos = clf.decision_function(X_test)
        prob_pos = (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())

        prob_cv = clf.decision_function(X_cv)
        prob_cv = (prob_cv - prob_cv.min()) / (prob_cv.max() - prob_cv.min())
    return prob_pos, prob_cv


def example_1():
    print("BEGIN")
    X, y = datasets.make_classification(
        n_samples=60000, n_features=20, n_informative=2, n_redundant=2
    )
    print("create_train_cv_test_split")
    X_train, X_cv, X_test, y_train, y_cv, y_test = create_train_cv_test_split(X, y)
    print("\t ... DONE")
    # kernel hyperparameter
    print("kernel hyp -- width of kernel")
    gamma = 1.0 / np.median(pairwise_distances(X_test, metric="euclidean")) ** 2
    print("\t ... DONE")
    # Create classifiers
    print("instantiate_classifiers")
    lr, gnb, svc, rfc = (
        instantiate_classifiers("lr"),
        instantiate_classifiers("gnb"),
        instantiate_classifiers("svc", C=1.0),
        instantiate_classifiers("rfc"),
    )
    print("\t ... DONE")

    # Instaniate Plot
    hist_data = []
    group_labels = []
    for clf, name in [
        (lr, "Logistic"),
        (gnb, "Naive_Bayes"),
        # (svc, "Support_Vector_Classification"),
        (rfc, "Random_Forest"),
    ]:
        print(f"CLF = {name}")
        fig = go.Figure()

        clf.fit(X_train, y_train)
        prob_pos, prob_cv = predict_probability(clf, X_test, X_cv)

        hist_data.append(prob_pos)
        group_labels.append(name)

        print("\t ... ELCE2")
        ELCE2_ = ELCE2(
            X_test,
            y_test,
            prob_pos,
            prob_kernel_width=0.1,
            kernel_function="rbf",
            gamma=gamma,
        )
        print("\t\t ... DONE")
        fig = plot_probability_frequency(prob_pos, ELCE2_, name)
        fig.write_image(f"output/ex1_{name}.png")
        # disp = CalibrationDisplay.from_estimator(clf, X_test, y_test)
        # plt.savefig(f"output/qq_{name}.png")

    fig = ff.create_distplot(hist_data, group_labels, bin_size=0.05, show_rug=False)
    fig.update_xaxes(range=[0, 1])
    fig.update_layout(title_text="Model Classifier Histograms")
    fig.write_image("output/ex1.png")


example_1()
