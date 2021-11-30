'''
The feature extraction model trained wrt Dtt and Dxx terms
The idea is presented in http://sip.unige.ch/articles/2021/Taran_WIFS2021.pdf
Author: Olga TARAN, University of Geneva, 2021
'''

from __future__ import print_function
import argparse
import yaml

import sys
sys.path.insert(0, '..')

import libs.yaml_utils as yaml_utils
from libs.utils import *

from libs.DataSetLoader import DataSetLoader
from libs.EstimatiorModel import TemplateEstimatior

# ======================================================================================================================
parser = argparse.ArgumentParser(description="TEST: the feature extraction model trained wrt Dtt and Dxx terms")
parser.add_argument("--config_path", default="./configuration.yml", type=str, help="The config file path")
# datset parameters
parser.add_argument("--image_type", default="rgb", type=str, choices=["rgb", "gray"], help="The image type")
parser.add_argument("--subset", default="test", type=str, choices=["train", "test", "validation"],
                    help="The train subset is used for the OC-SVM training/testing")
parser.add_argument("--data_paths", default=["../data/original/rgb/",
                                             "../data/fakes_1/paper_white/rgb/",
                                             "../data/fakes_1/paper_gray/rgb/",
                                             "../data/fakes_2/paper_white/rgb/",
                                             "../data/fakes_2/paper_gray/rgb/"
                                            ], type=str, help="The data paths")
# model parameters
parser.add_argument("--type", default="Dtt_Dxx", type=str, choices=["Dtt_Dxx", "Dtt_Dt_Dxx_Dx"], help="The trained model type")
parser.add_argument("--epoch", default=100, type=int, help="The test epoch")
parser.add_argument("--thr", default=0.5, type=float, help="The binarization threshold")
# visualisation parameters
parser.add_argument("--legend", default=["original",
                                         "fakes #1 white",
                                         "fakes #1 gray",
                                         "fakes #2 white",
                                         "fakes #2 gray",
                                         ], type=str, help="The legend for the visualisation")
# log mode
parser.add_argument("--is_debug", default=False, type=int, help="Is debug mode?")

args = parser.parse_args()
# ======================================================================================================================
set_log_config(args.is_debug)
log.info("PID = %d\n" % os.getpid())

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# ======================================================================================================================

def run(args):

    config = yaml_utils.Config(yaml.load(open(args.config_path)))
    symbol_size = config.dataset["args"]["symbol_size"]

    args.checkpoint_dir = "%s_%s" % (args.image_type, args.type)
    args.dir = "%s" % args.type
    args.save_suf = "%s_%s" % (args.image_type, args.type)

    log.info("Start Model preparation.....")
    Estimator = TemplateEstimatior(config, args, type=args.type)
    EstimationModel = Estimator.EstimationModel

    # === model scheme visualisation ===============================================================================
    if args.is_debug:
        log.info("UnetModel")
        Estimator.UnetModel.summary()

    # === Test =====================================================================================================
    EstimationModel.load_weights("%s/EstimationModel_epoch_%d" % (Estimator.checkpoint_dir, args.epoch))

    Dists   = []
    Indices = []
    for ind, path in enumerate(args.printed_paths):
        args.printed_path = path
        if len(Indices):
            config.dataset["args"]["test_indices"] = Indices

        DataGen = DataSetLoader(config, args, type=args.subset, is_debug_mode=args.is_debug)
        DataGen.initDataSet()

        batches = len(DataGen.getData()[0])
        if ind == 0 or True:
            Indices = DataGen.getIndices()

        Res = []
        l = -1
         for x_batch, y_batch in DataGen.datagen:
            l += 1
            if batches == l:
                break

            prediction = EstimationModel.predict(x_batch)

            t_predict = prediction[0]
            x_predict = prediction[1]

            t_predict = t_predict.reshape((256, 256))[:-1, :-1]
            x_predict = x_predict.reshape((256, 256, -1))[:-1, :-1]
            y_batch   = y_batch.reshape((256, 256))[:-1, :-1]
            x_batch   = x_batch.reshape((256, 256, -1))[:-1, :-1]

            t_predict_binary = postProcessingSimbolWise(np.copy(t_predict), symbol_size=symbol_size, thr=args.thr)

            dist_t = np.sum(np.logical_xor(y_batch.reshape((-1)), t_predict_binary.reshape((-1)))) / (symbol_size**2)
            dist_x = mse1D(x_batch.reshape((-1)), x_predict.reshape((-1)))
            Res.append([dist_t, dist_x])

        Dists.append(Res)

    saveListAsJson(Dists, "%s/%s_%s.txt" % (Estimator.results_dir, args.subset, args.save_suf))



# ======================================================================================================================
if __name__ == "__main__":
    run(args)









































