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
# import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import plotly.graph_objs as go
import plotly.express as px
from plotly import tools
from plotly.offline import iplot
import numpy as np


def create_train_cv_test_split(X, y, train_samples=20000, cv_samples=20000):
    X_train = X[:train_samples]
    X_cv = X[train_samples : train_samples + cv_samples]
    X_test = X[train_samples + cv_samples :]

    y_train = y[:train_samples]
    y_cv = y[train_samples : train_samples + cv_samples]
    y_test = y[train_samples + cv_samples :]
    return X_train, X_cv, X_test, y_train, y_cv, y_test

def instantiate_classifiers():
    lr = LogisticRegression()
    gnb = GaussianNB()
    svc = LinearSVC(C=1.0)
    rfc = RandomForestClassifier()
    return lr, gnb, svc, rfc


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
    X, y = datasets.make_classification(n_samples=60000, n_features=20, n_informative=2, n_redundant=2)
    print("create_train_cv_test_split")
    X_train, X_cv, X_test, y_train, y_cv, y_test = create_train_cv_test_split(X, y)
    print("\t ... DONE")
    # kernel hyperparameter
    print("kernel hyp")
    gamma = 1.0 / np.median(pairwise_distances(X_test, metric="euclidean")) ** 2
    print("\t ... DONE")
    # Create classifiers
    print("instantiate_classifiers")
    lr, gnb, svc, rfc = instantiate_classifiers()
    print("\t ... DONE")

    #Instaniate Plot
    hist_data = []
    group_labels = []
    for clf, name in [
        (lr, "Logistic"),
        (gnb, "Naive_Bayes"),
        (svc, "Support_Vector_Classification"),
        (rfc, "Random_Forest"),
    ]:
        print(f"CLF = {name}")
        fig = go.Figure()

        clf.fit(X_train, y_train)
        prob_pos, prob_cv = predict_probability(clf, X_test, X_cv)

        hist_data.append(prob_pos)
        group_labels.append(name)

        print(f"\t ... ELCE2")
        ELCE2_ = ELCE2(
            X_test,
            y_test,
            prob_pos,
            prob_kernel_width=0.1,
            kernel_function="rbf",
            gamma=gamma,
        )
        print(f"\t\t ... DONE")
        hist = go.Histogram(x=prob_pos, name=name)
        ELCE_trace = go.Scatter(x=[ELCE2_ * 100], y=[0,0], mode='markers', marker_size=20, name="ELCE2")
        fig = make_subplots(rows=2, cols=1, row_heights=[0.85, 0.15])
        fig.append_trace(hist, 1, 1)
        fig.append_trace(ELCE_trace, 2, 1)
        fig.update_layout(title_text=f'{name} Histogram')
        fig.write_html(f"ex1_{name}.html")

    fig = ff.create_distplot(hist_data, group_labels, bin_size=.05, show_rug=False)
    fig.update_xaxes(range=[0,1])
    fig.update_layout(title_text='Model Classifier Histograms')
    fig.write_html(f"ex1.html")


def example_2():
    print("BEGIN")
    X, y = datasets.make_classification(n_samples=60000, n_features=20, n_informative=2, n_redundant=2)
    print("create_train_cv_test_split")
    X_train, X_cv, X_test, y_train, y_cv, y_test = create_train_cv_test_split(X, y)
    print("\t ... DONE")
    # kernel hyperparameter
    print("kernel hyp")
    gamma = 1.0 / np.median(pairwise_distances(X_test, metric="euclidean")) ** 2
    print("\t ... DONE")
    # Create classifiers
    print("instantiate_classifiers")
    lr, gnb, svc, rfc = instantiate_classifiers()
    print("\t ... DONE")

    #Instaniate Plot
    for clf, name in [
        (lr, "Logistic"),
        (gnb, "Naive Bayes"),
        (svc, "Support Vector Classification"),
        (rfc, "Random Forest"),
    ]:
        print(f"CLF = {name}")
        plt, ax2, ax3 = instantiate_plot()
        clf.fit(X_train, y_train)
        prob_pos, prob_cv = predict_probability(clf, X_test, X_cv)
        ax2.hist(prob_pos, range=(0, 1), bins=20, label=name, histtype="step", lw=2)
        print(f"\t ... ELCE2")
        ELCE2_ = ELCE2(
            X_test,
            y_test,
            prob_pos,
            prob_kernel_width=0.1,
            kernel_function="rbf",
            gamma=gamma,
        )
        print(f"\t\t ... DONE")
        ax3.plot(ELCE2_ * 100, 1, "v", markersize=14, markeredgewidth=2)
        plt.savefig(f"ex1_{name}.png")


example_1()
