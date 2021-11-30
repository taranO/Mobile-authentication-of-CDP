'''
The OC-SVM training on the feature extracted based on the Dtt and Dxx terms
The idea is presented in http://sip.unige.ch/articles/2021/Taran_WIFS2021.pdf
Author: Olga TARAN, University of Geneva, 2021
'''

from __future__ import print_function
import argparse
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

import sys
sys.path.insert(0, '..')
from libs.utils import *

# ======================================================================================================================

parser = argparse.ArgumentParser(description="OC-SVM: on the feature extracted based on the Dtt and Dxx terms")
# datset parameters
parser.add_argument("--image_type", default="rgb", type=str, choices=["rgb", "gray"], help="The image type")
# model parameters
parser.add_argument("--type", default="Dtt_Dxx", type=str, choices=["Dtt_Dxx", "Dtt_Dt_Dxx_Dx"], help="The trained model type")
# visualisation parameters
parser.add_argument("--legend", default=["original",
                                         "fakes #1 white",
                                         "fakes #1 gray",
                                         "fakes #2 white",
                                         "fakes #2 gray",
                                         ], type=str, help="The legend for the visualisation")
# log mode
parser.add_argument("--is_debug", default=False, type=int, help="Debug mode")
args = parser.parse_args()

# ======================================================================================================================
set_log_config(args.is_debug)
log.info("PID = %d\n" % os.getpid())

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# ======================================================================================================================

def globalMinMax(Data):
    """Estimate the min and max value in the given list

    Parameters
    ----------
    Data: list of data

    """
    n = len(Data)
    global_min = []
    global_max = []
    for ii in range(n):
        global_min.append(np.asarray(Data[ii]).min())
        global_max.append(np.asarray(Data[ii]).max())

    return np.asarray(global_min).min(), np.asarray(global_max).max()

def make_meshgrid(x, y, h=.02):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 0.1, x.max() + 0.1
    y_min, y_max = y.min() - 0.1, y.max() + 0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)

    return out

# ======================================================================================================================

def run(args):

    save_to_ = "checkpoints/one_class_svm/%s_lr1e-4" % args.type
    makeDir(save_to_)
    result_dir = "results/%s" % args.type
    file_suf = "%s_%s" % (args.image_type, args.type)

    log.info("Train data preparation.....")
    file_train = "./anomaly_detection/%s/train_%s.txt" % (result_dir, file_suf)
    Dists = loadListFromJson(file_train)

    log.info("Test data preparation.....")
    file_test = "./anomaly_detection/%s/test_%s.txt" % (result_dir, file_suf)
    Dists_test = loadListFromJson(file_test)

    # === OC-SVM training ==============================================================================================
    X_train = np.asarray(Dists[0])
    clf = make_pipeline(StandardScaler(), svm.OneClassSVM(kernel="rbf", nu=0.0005, gamma=0.1))
    clf.fit(X_train)

    # === OC-SVM test ==================================================================================================
    fig, ax = plt.subplots()
    for ll in range(len(Dists_test)):
        if ll == 0:
            X0 = np.asarray(Dists_test[ll])[:, 0]
            X1 = np.asarray(Dists_test[ll])[:, 1]
        else:
            X0 = np.hstack((X0, np.asarray(Dists_test[ll])[:, 0]))
            X1 = np.hstack((X1, np.asarray(Dists_test[ll])[:, 1]))

    xx, yy = np.meshgrid(np.linspace(-50, 450, 100),
                         np.linspace(0.001, 0.003, 100))
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7), cmap=plt.cm.Purples)
    plt.contourf(xx, yy, Z, levels=[0, Z.max()], colors='palevioletred')

    title = ""
    for ll in range(len(Dists_test)):
        X = np.asarray(Dists_test[ll])
        m = X.shape[0]
        if ll == 0:
            Y = np.ones((1, m))
        else:
            Y = -1 + np.zeros((1, m))

        Y = Y.reshape((-1))

        acc = accuracy_score(Y, clf.predict(X), normalize=False)
        error = Y.size - acc
        title += "%s: errors = %d (%0.3f)\n" % (args.legend[ll], error, error / Y.size)

        ax.scatter(X[:, 0], X[:, 1], s=10, label=args.legend[ll])

    log.info(title)
    ax.legend(fontsize=18,  markerscale=2,)
    plt.grid()
    plt.xlabel("$d_{Hamming}(t, \hat{t})$", fontsize=22)
    plt.ylabel("$d_{l_2}(x, \hat{x})$", fontsize=22)
    plt.tick_params(axis='both', labelsize=18)
    fig.tight_layout()

    plt.savefig("%s/oc-svm_%s.pdf" % (result_dir, file_suf))
    plt.close()

# ======================================================================================================================
if __name__ == "__main__":
    run(args)
