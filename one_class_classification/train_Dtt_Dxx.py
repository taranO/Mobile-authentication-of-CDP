'''
The feature extraction model trained wrt Dtt and Dxx terms
The idea is presented in http://sip.unige.ch/articles/2021/Taran_WIFS2021.pdf
Author: Olga TARAN, University of Geneva, 2021
'''

from __future__ import print_function
import argparse
import yaml

import sys
sys.path.insert(0,'..')

import libs._yaml_utils as yaml_utils
from libs._utils import *

from libs.DataSetLoader import DataSetLoader
from libs.EstimatiorModel import TemplateEstimatior

# ======================================================================================================================
parser = argparse.ArgumentParser(description="TRAIN: the feature extraction model trained wrt Dtt and Dxx terms")
parser.add_argument("--config_path", default="./configuration.yml", type=str, help="The config file path")
# datset parameters
parser.add_argument("--image_type", default="rgb", type=str, choices=['rgb', 'gray'], help="The image type")
# model parameters
parser.add_argument("--type", default="Dtt_Dxx", type=str, help="The trained model type")
parser.add_argument("--epochs", default=100, type=int, help="Number of training epochs")
parser.add_argument("--start_epoch", default=0, type=int, help="The start epoch")
# log mode
parser.add_argument("--is_debug", default=True, type=int, help="Is debug mode?")

args = parser.parse_args()
# ======================================================================================================================

set_log_config(args.is_debug)
log.info("PID = %d\n" % os.getpid())

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# ======================================================================================================================

def train(args):

    config = yaml_utils.Config(yaml.load(open(args.config_path)))

    args.checkpoint_dir = "%s_%s" % (args.image_type, args.type)
    args.dir = "%s" % args.type

    log.info("Start Model preparation.....")
    Estimator = TemplateEstimatior(config, args, type=args.type)
    EstimationModel = Estimator.EstimationModel

    # --- Data set -----------
    log.info("Start Train Data loading.....")
    DataGenTrain = DataSetLoader(config, args, type="train", is_debug_mode=args.is_debug)
    DataGenTrain.initDataSet()

    # === model scheme visualisation ===============================================================================
    if args.is_debug:
        log.info("UnetXModel")
        Estimator.UnetXModel.summary()

        log.info("UnetTModel")
        Estimator.UnetTModel.summary()

    # === Training =================================================================================================
    if args.start_epoch > 0:
        EstimationModel.load_weights("%s/EstimationModel_epoch_%d" % (Estimator.checkpoint_dir, args.start_epoch))

    for epoch in range(args.start_epoch+1, args.epochs + 1):
        Loss_x = []
        Loss_t  = []
        batches = 0
        save_each = saveSpeed(epoch)
        for x_batch, y_batch in DataGenTrain.datagen:

            y_batch = y_batch.reshape((-1, config["dataset"]["args"]["target_size"][0],
                                       config["dataset"]["args"]["target_size"][1], 1))
            # --- estimator -----
            loss = EstimationModel.train_on_batch(x_batch, [y_batch, x_batch])
            Loss_t.append(loss[1])
            Loss_x.append(loss[2])

            batches += 1
            if batches >= DataGenTrain.n_batches:
                # we need to break the loop by hand because
                # the generator loops indefinitely
                break

        log.info(f"epoch : {epoch}, \t"
                 f"mse_t = {np.mean(np.asarray(Loss_t))}\t "
                 f"mse_x = {np.mean(np.asarray(Loss_x))}")

        # ------------------------------------------------------------------------
        if epoch % save_each == 0 or epoch == args.epochs:
            EstimationModel.save_weights("%s/EstimationModel_epoch_%d" % (Estimator.checkpoint_dir, epoch))

# ======================================================================================================================
if __name__ == "__main__":
    train(args)



























































