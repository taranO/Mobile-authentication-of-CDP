'''
The OC-SVM training on the feature extracted based on the Dtt, Dt, Dxx and Dx terms
The idea is presented in http://sip.unige.ch/articles/2021/Taran_WIFS2021.pdf
Author: Olga TARAN, University of Geneva, 2021
'''

from __future__ import print_function
import argparse
from sklearn import svm
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

import sys
sys.path.insert(0,'..')

from libs.utils import *

# ======================================================================================================================

parser = argparse.ArgumentParser(description="OC-SVM: on the feature extracted based on the Dtt, Dt, Dxx and Dx terms")

# datset parameters
parser.add_argument("--image_type", default="rgb", type=str, choices=["rgb", "gray"], help="The image type")
# model parameters
parser.add_argument("--type", default="Dtt_Dt_Dxx_Dx", type=str, help="The trained model type")
parser.add_argument("--w", default="0.01", type=str, help="Weight parameter")
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

def run(args):

    dir_ = "results/%s" % args.type
    file_suf = "%s_%s" % (args.image_type, args.type)

    # === OC-SVM training ==============================================================================================
    file = "%s/%s_%s_w%s.txt" % (dir_, "train", file_suf, args.w)
    ResultsX = loadListFromJson(file)
    X_train = np.asarray(ResultsX[0][0])[:, 0:2] # 0:2, [1, 3]

    clf = make_pipeline(StandardScaler(), svm.OneClassSVM(kernel="rbf", nu=0.0005, gamma=0.1))
    clf.fit(X_train)

    # === OC-SVM test ==================================================================================================
    file = "%s/%s_%s_w%s.txt" % (dir_, "test", file_suf, args.w)
    ResultsY = loadListFromJson(file)
    Dists_test = ResultsY[0]

    fig, ax = plt.subplots()
    xx, yy = np.meshgrid(np.linspace(-50, 500, 100),
                         np.linspace(0.0008, 0.0032, 100))
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7), cmap=plt.cm.Purples)
    plt.contourf(xx, yy, Z, levels=[0, Z.max()], colors='palevioletred')

    title = ""
    for ll in range(len(Dists_test)):
        X = np.asarray(Dists_test[ll])[:, 0:2] # 0 - Dtt, 1 - Dxx, 2 - Dt, 3 - Dx
        m = X.shape[0]
        if ll == 0:
            Y = np.ones((1, m))
        else:
            Y = -1 + np.zeros((1, m))

        Y = Y.reshape((-1))

        acc = accuracy_score(Y, clf.predict(X), normalize=False)
        error = Y.size - acc
        title += "%s: errors = %d (%0.5f, %0.5f)\n" % (args.legend[ll], error, error / Y.size, 100*error / Y.size)

        ax.scatter(X[:, 0], X[:, 1], s=10, label=args.legend[ll])

    log.info(title)

    ax.legend(fontsize=18,  markerscale=2,)
    plt.grid()
    plt.xlabel("$d_{Hamming}(t, \hat{t})$", fontsize=22)
    plt.ylabel("$d_{l_2}(x, \hat{x})$", fontsize=22)
    plt.tick_params(axis='both', labelsize=18)
    fig.tight_layout()

    plt.savefig('%s/oc-svm_%s_w%s.pdf' % (dir_, file_suf, args.w))
    plt.close()


# ======================================================================================================================
if __name__ == "__main__":
    run(args)













