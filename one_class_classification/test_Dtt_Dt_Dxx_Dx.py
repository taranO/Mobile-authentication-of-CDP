'''
The feature extraction model trained wrt Dtt, Dt, Dxx and Dx terms
The idea is presented in http://sip.unige.ch/articles/2021/Taran_WIFS2021.pdf
Author: Olga TARAN, University of Geneva, 2021
'''

from __future__ import print_function
from tensorflow.keras.utils import plot_model

import argparse
import datetime
import yaml
from scipy.spatial.distance import hamming
from skimage import filters
from skimage.metrics import mean_squared_error as mse

import sys
sys.path.insert(0,'..')

import libs.yaml_utils as yaml_utils
from libs.utils import *

from libs.DataSetLoader import DataSetLoader
from libs.EstimatiorModel import TemplateEstimatior

# ======================================================================================================================
parser = argparse.ArgumentParser(description="TEST: the feature extraction model trained wrt Dtt, Dt, Dxx and Dx terms")
parser.add_argument("--config_path", default="./configuration.yml", type=str, help="The config file path")
# datset parameters
parser.add_argument("--image_type", default="rgb", type=str, choices=["rgb", "gray"], help="The image type")
parser.add_argument("--printed_paths", default=["../data/original/rgb/",
                                             "../data/fakes_1/paper_white/rgb/",
                                             "../data/fakes_1/paper_gray/rgb/",
                                             "../data/fakes_2/paper_white/rgb/",
                                             "../data/fakes_2/paper_gray/rgb/"
                                            ], type=str, help="The data paths")
# model parameters
parser.add_argument("--type", default="Dtt_Dt_Dxx_Dx", type=str, help="The trained model type")
parser.add_argument("--subset", default="validation", type=str, choices=["train", "test", "validation"],
                    help="The train subset is used for the OC-SVM training/testing")
parser.add_argument("--w", default="0.01", type=str, help="Weight parameter")
parser.add_argument("--epoch", default=100, type=int, help="The test epoch")

parser.add_argument("--thr", default=0.5, type=float, help="Binarization threshold")
parser.add_argument("--metrics", default=["pearson", "xor_otsu", "mse", "l1"], type=str, help="The metrics for cross-validation")
# visualisation
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

def threshold(data, thr):
    """Input data binarization wrt given threshold

    Args:
        data: data to binarize
        thr: binarization threshold

    Returns:
        np.ndarray: binarized data"""

    data[data <= thr] = 0
    data[data != 0] = 1

    return data

def applyMetric(template_code, printed_code, metric, thr=0.5):

    if metric == "pearson":
        dist, _ = scipy.stats.pearsonr(template_code.reshape(-1), printed_code.reshape(-1))

    elif metric == "l1":
        diff = printed_code - template_code
        dist = np.sum(abs(diff.reshape((-1)))) /printed_code.size

    elif metric == "mse":
        dist = mse(template_code, printed_code)

    elif metric == "xor":
        if not args.is_dr_normalization:
            template_code = normaliseDR(template_code)
            printed_code = normaliseDR(printed_code)
        dist = hamming(threshold(template_code, thr).reshape((-1)), threshold(printed_code, thr).reshape((-1)))

    return dist
# ======================================================================================================================
def run(args)

    config = yaml_utils.Config(yaml.load(open(args.config_path)))
    symbol_size = config.dataset["args"]["symbol_size"]

    args.checkpoint_dir = "%s_%s_w%s" % (args.image_type, args.type, args.w)
    args.dir = "%s" % args.type
    args.save_suf = "%s_%s" % (args.image_type, args.type)

    log.info("Start Model preparation.....")
    Estimator = TemplateEstimatior(config, args, type=args.type)
    EstimationModel = Estimator.EstimationModel

    # === model scheme visualisation ===============================================================================
    if args.is_debug:
        log.info("UnetModel")
        Estimator.UnetModel.summary()

    # === Test =================================================================================================
    Indices = []
    Dists   = []
    Labels  = []

    EstimationModel.load_weights("%s/EstimationModel_epoch_%d" % (Estimator.checkpoint_dir, args.epoch))

    for ind, path in enumerate(args.printed_paths):
        Dist = []

        args.printed_path = path
        if len(Indices):
            name = "%s_indices" % args.subset
            config.dataset["args"][name] = Indices

        DataGenerator = DataSetLoader(config, args, type=args.subset, is_debug_mode=args.is_debug)
        DataGenerator.initDataSet()
        batches = DataGenerator.n_batches
        if ind == 0:
            Indices = DataGenerator.getIndices()

        loss_mse = []
        loss_xor = []
        loss_dt  = []
        loss_dx = []
        for x_batch, y_batch in DataGenerator.datagen:
            batches -= 1
            if batches < 0:
                break

            prediction = EstimationModel.predict(x_batch)

            t_predict = prediction[0]
            x_predict = prediction[1]

            dx_predict = (prediction[2][0][0]).astype(np.float64)
            dt_predict = (prediction[3][0][0]).astype(np.float64)

            t_predict = t_predict.reshape((256, 256))[:-1, :-1]
            x_predict = x_predict.reshape((256, 256, -1))[:-1, :-1]
            y_batch   = y_batch.reshape((256, 256))[:-1, :-1]
            x_batch   = x_batch.reshape((256, 256, -1))[:-1, :-1]

            t_predict_binary = postProcessingSimbolWise(np.copy(t_predict), symbol_size=symbol_size, thr=args.thr)

            dist_t = np.sum(np.logical_xor(y_batch.reshape((-1)), t_predict_binary.reshape((-1)))) / (symbol_size**2)
            dist_x = mse1D(x_batch.reshape((-1)), x_predict.reshape((-1)))

            loss_mse.append(dist_x)
            loss_xor.append(dist_t)
            loss_dt.append(dt_predict)
            loss_dx.append(dx_predict)

            if len(args.metrics) > 0:
                M = [dist_t, dist_x, dt_predict, dx_predict]
                for metric in args.metrics:
                    M.append(applyMetric(np.copy(y_batch), np.copy(x_batch), metric))
                Dist.append(M)
            else:
                Dist.append([dist_t, dist_x, dt_predict, dx_predict])
            Labels.append(ind)
        Dists.append(Dist)

        if ind == 0:
            log.info("epoch %s:\t mse = %0.5f,\t xor = %0.5f,\t dx  = %0.5f,\t dt = %0.5f" % (epoch, np.mean(np.asarray(loss_mse)),
                                                                                           np.mean(np.asarray(loss_xor)),
                                                                                           np.mean(np.asarray(loss_dx)),
                                                                                           np.mean(np.asarray(loss_dt))))

    saveListAsJson([Dists, Labels], "%s/%s_%s_w%s_multi-metrics.txt" % (Estimator.results_dir, args.subset, args.save_suf, args.w))


# ======================================================================================================================
if __name__ == "__main__":
    run(args)








































