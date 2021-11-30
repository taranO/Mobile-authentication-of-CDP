'''
Supervised classification of CDP
Author: Olga TARAN, University of Geneva, 2021
'''

from __future__ import print_function
import argparse
import datetime
import yaml

import sys
sys.path.insert(0,'..')

import libs.yaml_utils as yaml_utils
from libs.utils import *

from libs.ClassifierDataLoader import ClassifierDataLoader
from libs.ClassificationModel import ClassificationModel

# ======================================================================================================================
parser = argparse.ArgumentParser(description="TEST: the supervised classification of CDP")
parser.add_argument("--config_path", default="./configuration.yml", type=str, help="The config file path")
# datset parameters
parser.add_argument("--image_type", default="rgb", type=str, choices=['rgb', 'gray'], help="The image type")
parser.add_argument("--data_paths", default=["../data/original/rgb/",
                                             "../data/fakes_1/paper_white/rgb/",
                                             "../data/fakes_1/paper_gray/rgb/",
                                             "../data/fakes_2/paper_white/rgb/",
                                             "../data/fakes_2/paper_gray/rgb/"
                                            ], type=str, help="The data paths")

# model parameters
parser.add_argument("--lr", default=1e-4, type=str, help="Training learning rate")
parser.add_argument("--epoch", default=50, type=int, help="The test epoch")
parser.add_argument("--n_classes", default=5, type=int, choices=[2, 5], help="The number classes: 2 - original or fakes, 5 - original and 4 types of fakes")
parser.add_argument("--is_max_pool", default=True, type=int, help="Is to use max pooling in the trained model?")
# log mode
parser.add_argument("--is_debug", default=False, type=int, help="Is debug mode?")

args = parser.parse_args()

# ======================================================================================================================
set_log_config(args.is_debug)
log.info("PID = %d\n" % os.getpid())

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# ======================================================================================================================
def run():
    config = yaml_utils.Config(yaml.load(open(args.config_path)))

    args.checkpoint_dir = "%s_supervised_classifier_n_%d" % (args.image_type, args.n_classes)
    args.dir = "supervised_classifier_lr%s" % args.lr

    log.info("Model preparation.....")
    model = ClassificationModel(config, args)
    Classifier = model.ClassifierModel
    log.info("Pretrained model loading.....")
    Classifier.load_weights("%s/Classifier_epoch_%d" % (model.checkpoint_dir, args.epoch))

    log.info("Data loading.....")
    DataGen = ClassifierDataLoader(config, args, type="test", is_debug_mode=args.is_debug)
    DataGen.initDataSet()
    n_batches = DataGen.n_batches

    # === model scheme visualisation ===============================================================================
    if args.is_debug:
        log.info("EstimationModel")
        model.Classifier.summary()

    # === Test =================================================================================================
    errors_1 = np.zeros((2))
    counts_1 = np.zeros((2))

    errors_2 = np.zeros((3))
    counts_2 = np.zeros((3))

    errors_3 = np.zeros((args.n_classes))
    counts_3  = np.zeros((args.n_classes))

    l = -1
    for x_batch, labels in DataGen.datagen:
        l += 1
        if  l >= n_batches:
            break

        prediction = Classifier.predict(x_batch)

        true_lab = np.argmax(labels)
        pred_lab = np.argmax(prediction)

        # --------------------------------------------
        # 2 classes: original, fakes
        if true_lab == 0 and pred_lab != 0:
            errors_1[0] += 1
        elif true_lab != 0 and pred_lab == 0:
            errors_1[1] += 1

        # --------------------------------------------
        # 3 classes: original, fakes#1, fakes#2
        if true_lab == 0 and pred_lab != 0:
            errors_2[0] += 1
        elif true_lab in [1, 2] and pred_lab not in [1, 2]:
            errors_2[1] += 1
        elif true_lab in [3, 4] and pred_lab not in [3, 4]:
            errors_2[2] += 1

        # --------------------------------------------
        # 5 classes: original; fakes#1 white, gray; fakes#2 white, gray
        if true_lab != pred_lab:
            errors_3[true_lab] += 1

        if true_lab == 0:
            counts_1[0] += 1
            counts_2[0] += 1
        elif true_lab in [1, 2]:
            counts_1[1] += 1
            counts_2[1] += 1
        elif true_lab in [3, 4]:
            counts_1[1] += 1
            counts_2[2] += 1
        counts_3[true_lab] += 1

    log.info("Binary _classification:")
    log.info(f" - errors: {errors_1}\t total error: {np.sum(errors_1)}")
    errors = np.divide(errors_1, counts_1)
    log.info(f" - errors: {errors}\t average error: {np.sum(errors_1)/n_batches}")

    log.info("3 classes: original, fakes#1, fakes#2:")
    log.info(f" - errors: {errors_2}\t total error: {np.sum(errors_2)}")
    errors = np.divide(errors_2, counts_2)
    log.info(f" - errors: {errors}\t average error: {np.sum(errors_2)/n_batches}")

    log.info("5 classes: original; fakes#1 white, gray; fakes#2 white, gray")
    log.info(f" - errors: {errors_3}\t total error: {np.sum(errors_3)}")
    errors = np.divide(errors_3, counts_3)
    log.info(f" - errors: {errors}\t average error: {np.sum(errors_3)/n_batches}")

# ======================================================================================================================
if __name__ == "__main__":
    run()

















































