'''
Supervised classification of CDP
Author: Olga TARAN, University of Geneva, 2021
'''

from __future__ import print_function

import argparse
import yaml
import sys
sys.path.insert(0,'..')

import libs.yaml_utils as yaml_utils
from libs.utils import *

from libs.ClassifierDataLoader import ClassifierDataLoader
from libs.ClassificationModel import ClassificationModel

# ======================================================================================================================
parser = argparse.ArgumentParser(description="TRAIN: the supervised classification of CDP")
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
parser.add_argument("--lr", default=1e-4, type=float, help="Training learning rate")
parser.add_argument("--epochs", default=100, type=int, help="Number of training epochs")
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

def train():
    config = yaml_utils.Config(yaml.load(open(args.config_path)))

    args.checkpoint_dir = "%s_supervised_classifier_n_%d" % (args.image_type, args.n_classes)
    args.dir = "supervised_classifier_lr%s" % args.lr

    log.info("Start Model preparation.....")
    model = ClassificationModel(config, args)
    Classifier = model.ClassifierModel

    log.info("Start Train Data loading.....")
    DataGenTrain = ClassifierDataLoader(config, args, type="train", is_debug_mode=args.is_debug)
    DataGenTrain.initDataSet()


    # === model scheme visualisation ===============================================================================
    if args.is_debug:
        log.info("EstimationModel")
        model.Classifier.summary()

    # === Training =================================================================================================
    for epoch in range(args.epochs):
        Loss = []
        batches = 0
        save_each = saveSpeed(epoch)        
        for x_batch, labels in DataGenTrain.datagen:
            loss = Classifier.train_on_batch(x_batch, labels)
            Loss.append(loss)
            batches += 1
            if batches >= DataGenTrain.n_batches:
                # we need to break the loop by hand because
                # the generator loops indefinitely
                break

        log.info(f"epoch : {epoch}, \t loss = {np.mean(np.asarray(Loss))}")

        # ------------------------------------------------------------------------
        if epoch % save_each == 0 or epoch == args.epochs:
            Classifier.save_weights("%s/Classifier_epoch_%d" % (model.checkpoint_dir, epoch))

# ======================================================================================================================
if __name__ == "__main__":
    train()
















































