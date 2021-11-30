'''
Supervised _classification of CDP
Author: Olga TARAN, University of Geneva, 2021
'''

from __future__ import print_function
from tensorflow.keras.models import Model
from sklearn.manifold import TSNE
import argparse
import yaml

import sys
sys.path.insert(0,'..')

import libs.yaml_utils as yaml_utils
from libs.utils import *

from libs.ClassifierDataLoader import ClassifierDataLoader
from libs.ClassificationModel import ClassificationModel

# ======================================================================================================================
parser = argparse.ArgumentParser(description="T-SNE: the supervised classification of CDP")
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
parser.add_argument("--epoch", default=8, type=int, help="The test epoch")
parser.add_argument("--n_classes", default=5, type=int, help="...")
parser.add_argument("--is_max_pool", default=True, type=int, help="Is to use max pooling in the trained model?")
parser.add_argument("--layer", default="dense_4", type=str, help="The name of layer used for the T-SNE.")
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

    log.info("Data loading.....")
    args.n_classes = 5
    DataGen = ClassifierDataLoader(config, args, type="test", is_debug_mode=args.is_debug)
    DataGen.initDataSet()

    # === model scheme visualisation ===========================================================================
    if args.is_debug:
        log.info("Classifier")
        model.Classifier.summary()

    # === Test =================================================================================================
    Classifier.load_weights("%s/Classifier_epoch_%d" % (model.checkpoint_dir, args.epoch))
    LSModel = Model(inputs=Classifier.input,
                    outputs=Classifier.get_layer('Classifier').get_layer(args.layer).output)
    Data = []
    Labels = []

    l = -1
    n_batches = DataGen.n_batches
    for x_batch, labels in DataGen.datagen:
        l += 1
        if l >= n_batches:
            break

        laten_space = LSModel.predict(x_batch)
        Data.append(laten_space.reshape((-1)))
        Labels.append(np.argmax(labels))

    # === TSNE visualisation ==========================================================================================
    Data = np.asarray(Data)
    Labels = np.asarray(Labels)

    labels = np.unique(Labels)
    log.info(f"Data.shape: {Data.shape}, \t Labels.shape: {Labels.shape}, \t unique labels: {labels}")

    tsne_model = TSNE(n_components=2, random_state=0)
    Data_embeded = tsne_model.fit_transform(Data)

    fig = plt.figure()
    for i in labels:
        plt.plot(Data_embeded[Labels == i, 0], Data_embeded[Labels == i, 1], ".")

    plt.tick_params(axis='both', labelsize=16)
    plt.legend(args.legend[0:len(labels)], fontsize=18, markerscale=2, loc=4)
    plt.grid()
    plt.savefig('%s/tsne_%s_latent_space_%s_lr1e-4_epoch%d.pdf' % (model.results_dir,
                                                                   args.image_type, args.layer, args.epoch))
    plt.close()

# ======================================================================================================================
if __name__ == "__main__":
    run()













































