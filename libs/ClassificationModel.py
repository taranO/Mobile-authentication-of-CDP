import tensorflow as tf
import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
import datetime

from libs.to_public.BaseClass import BaseClass
from libs.Classifier import Classifier
# ======================================================================================================================

class ClassificationModel(BaseClass):

    def __init__(self, config, args):

        self.config = config
        self.type = type
        self.__layer_normalisation = args._layer_normalisation if "layer_normalisation" in args else self.config.models["classifier"]["layer_normalisation"]
        self.__max_pool = args.is_max_pool if "is_max_pool" in args else False
        self.n_classes = args.n_classes if "n_classes" in args else self.config.models["classifier"]["n_classes"]
        self.__kernel_size = args._kernel_size if "kernel_size" in args else 5
        self.__stride = args._stride if "stride" in args else 1
        self.__lr = args.lr if "lr" in args else config.config.models["classifier"]["lr"]
        self.__createResDirs(args)

        self.tensor_board = keras.callbacks.TensorBoard(
            log_dir=self.tensor_board_dir,
            histogram_freq=1,
            write_graph=True,
            write_grads=True,
            write_images=True
        )

        self.__initClassificationModel(args)

        self.tensor_board.set_model(self.ClassifierModel)


    def __initClassificationModel(self):

        input  = Input(shape=(self.config["models"]["classifier"]["target_size"][0],
                              self.config["models"]["classifier"]["target_size"][1],
                              self.config["models"]["classifier"]["target_size"][2]))

        self.Classifier = Classifier(conv_filters=self.config.models["classifier"]["conv_filters"],
                                     dense_filters=self.config.models["classifier"]["dense_filters"],
                                     n_classes=self.n_classes,
                                     kernel_size=self.__kernel_size,
                                     stride=self.__stride,
                                     max_pool=self.__max_pool,
                                     layer_normalisation=self.__layer_normalisation).init(input)
        self.ClassifierModel = Model(input, self.Classifier(input))

        optimizer = self.__getOptimizer(self.config.models["classifier"]["optimizer"], self.__lr)
        loss = self.__getLoss(self.config.models["classifier"]["loss"])

        self.ClassifierModel.compile(loss=loss, optimizer=optimizer)


    def __getLoss(self, loss):

        if loss == "binary_crossentropy":
            return tf.keras.losses.binary_crossentropy
        elif loss == "mse":
            return tf.keras.losses.mean_squared_error

    def __getOptimizer(self, optimazer, lr):

        if optimazer == "adam":
            return tf.keras.optimizers.Adam(learning_rate=lr)


    def __createResDirs(self, args):

        self.checkpoint_dir = self.makeDir(self.config.checkpoint_dir + "/" + args.checkpoint_dir)
        self.results_dir = self.makeDir(self.config.results_dir + "/" + args.dir)
        self.tensor_board_dir = self.makeDir("./TensorBoard/" + args.dir + "/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        self.makeDir(self.config.results_dir + "/" + args.dir + "/prediction/")


    def printLog(self, args):
        info = {"checkpoint_dir": self.checkpoint_dir,
                "results_dir": self.results_dir,
                "classifier": self.config.models['classifier'],
                "batch_size": self.config.batchsize,
                "epochs": args.epochs if "epochs" in args else args.epoch,
                }
        info["lr"] = self.__lr
        info["kernel_size"] = self.__kernel_size
        info["stride"] = self.__stride
        info["n_classes"] = self.n_classes

        self.printToLog(info)