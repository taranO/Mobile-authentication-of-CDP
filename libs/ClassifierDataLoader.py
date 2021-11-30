import os
import math
import numpy as np

import skimage.io
from skimage.transform import rotate
from skimage.exposure import adjust_gamma
from sklearn.model_selection import train_test_split

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from libs.to_public.BaseClass import BaseClass

# ======================================================================================================================

class ClassifierDataLoader(BaseClass):

    def __init__(self, config, args, type="train", is_debug_mode=True):

        self.__is_debug_mode = is_debug_mode
        self.config = config
        self.type   = type if type else "test"
        self.symbol_size = config.dataset['args']["symbol_size"]

        self.__seed = args.seed if "seed" in args else -1
        self.__indices = []

        self.__binary_codes_path = args.templates_path if "templates_path" in args else config.dataset["args"]["templates_path"]
        self.__data_paths = args.data_paths
        self.n_classes = args.n_classes if "n_classes" in args else self.config.models["classifier"]["n_classes"]

        self.n_batches = 0
        self.__data    = []
        self.__labels  = []


    def initDataSet(self):
        # load data
        self.__loadData(self.config.dataset['args'])

        self.__data = self.reshapeData(self.__data, self.config.models['classifier']["target_size"])

        # create data generator
        datagen = ImageDataGenerator(samplewise_center=False, samplewise_std_normalization=False)

        # compile the data generators
        if self.type == "train":
            self.n_batches = self.__data.shape[0] // self.config.batchsize + 1
            self.datagen = datagen.flow(x=self.__data, y=self.__labels,
                                        batch_size=self.config.batchsize, shuffle=True)
        else:
            self.n_batches = self.__data.shape[0]
            self.datagen = datagen.flow(x=self.__data, y=self.__labels, batch_size=1, shuffle=False)

    def __loadData(self, args):

        list = os.listdir(self.__binary_codes_path)
        list.sort()

        if self.__seed >= 0:
            args["train_indices"], args["validation_indices"], args["test_indices"] = self.__getIndices(len(list), args)

        if self.type == "train":
            self.__data, self.__labels = self.__loadImages(list, args, args["train_indices"])
            # train data augmentation
            if args["augmentation"]:
                self.__augmentTrainData(args["augmentation_args"])
        elif self.type == "validation":
            self.__data, self.__labels = self.__loadImages(list, args, args["validation_indices"])
        if self.type == "test":
            self.__data, self.__labels = self.__loadImages(list, args, args["test_indices"])


    def __getIndices(self, N, args):
        np.random.seed(seed=self.__seed)
        indices = np.arange(1, N+1)

        # exclude the bad codes
        indices = np.setdiff1d(indices, np.asarray(args["bad_indices"]))

        train_indices, test_indices = train_test_split(indices, test_size=args["test_ration"], shuffle=True)
        train_indices, val_indices = train_test_split(train_indices, test_size=args["val_ratio"], shuffle=True)

        return train_indices, val_indices, test_indices

    def __loadImages(self, list, args, inds):

        self.__indices = inds if not isinstance(inds, str) else self.__loadIndices(inds)

        M = len(self.__data_paths)
        N = len(self.__indices)
        data   = []
        labels = []

        for i, ind in enumerate(self.__indices):
            for j in range(M):
                # load data
                image_x = skimage.io.imread(self.__data_paths[j] + "/" + list[ind-1]).astype(np.float64)
                image_x = self.__centralCrop(image_x, targen_size=self.config.models['classifier']["target_size"])
                data.append(self.normaliseDynamicRange(image_x, args))

                if self.n_classes == 1:
                    if j == 0:
                        labels.append(0)
                    else:
                        labels.append(1)
                elif self.n_classes != M:
                    onehot_label = [0 for _ in range(self.n_classes)]
                    if j == 0:
                        onehot_label[0] = 1
                    else:
                        onehot_label[1] = 1

                    labels.append(onehot_label)
                else:
                    onehot_label = [0 for _ in range(M)]
                    onehot_label[j] = 1

                    labels.append(onehot_label)

        return np.asarray(data), np.asarray(labels)

    def __centralCrop(self, image, targen_size=[330, 330, 1]):
        if image.shape[0] <= targen_size[0] and image.shape[1] <= targen_size[1]:
            return image

        height, width = image.shape[0:2]
        top_corner = self.symbol_size*math.floor((height // 2 - targen_size[0] //2) / self.symbol_size)
        left_corner = self.symbol_size*math.floor((width // 2 - targen_size[1] //2) / self.symbol_size)

        return image[top_corner:top_corner+targen_size[0], left_corner:left_corner+targen_size[1]].reshape(targen_size)

    def __augmentTrainData(self, args):
        data   = self.__data
        labels = self.__labels

        for a in args["first_order"]:
            if a == "rotation":
                augmented_data, augmented_labels = self.__rotateData(data, labels, args)
            elif a == "flip":
                augmented_data, augmented_labels = self.__flipData(data, labels, args)
            elif a == "gamma":
                augmented_data, augmented_labels = self.__adjustGamma(data, labels, args)

            if len(self.__data):
                self.__data = np.concatenate((self.__data, augmented_data), axis=0)
                self.__labels  = np.concatenate((self.__labels, augmented_labels), axis=0)
            else:
                self.__data = augmented_data
                self.__labels  = augmented_labels


    def __rotateData(self, data, labels, args, order=1):
        rotated_data   = []
        rotated_labels = []

        angels = args["rotation_angles"]

        n = len(data)
        for ind in range(n):
            for r in angels:
                image_rx = rotate(data[ind], r, resize=False)

                if order != 2 and len(args["second_order"]):
                    augmented_data   = []
                    augmented_labels = []
                    for a in args["second_order"]:
                        if a == "flip":
                            augmented_data, augmented_labels = self.__flipData([image_rx], [labels[ind]], args, order=2)
                        elif a == "gamma":
                            augmented_data, augmented_labels = self.__adjustGamma([image_rx], [labels[ind]], args, order=2)

                        if len(rotated_data):
                            rotated_data = np.concatenate((rotated_data, augmented_data),axis=0)
                            rotated_labels = np.concatenate((rotated_labels, augmented_labels),axis=0)
                        else:
                            rotated_data = augmented_data
                            rotated_labels  = augmented_labels

                else:
                    rotated_data.append(image_rx)
                    rotated_labels.append(labels[ind])

        return rotated_data, rotated_labels



    def __adjustGamma(self, data, labels, args, order=1):
        adjusted_data = []
        adjusted_labels = []

        gamma = np.arange(args["gamma"][0], args["gamma"][1]+args["gamma"][2], args["gamma"][2])

        n = len(data)
        for ind in range(n):
            for g in gamma:
                image_gx = adjust_gamma(data[ind], gamma=g)

                if order != 2 and len(args["second_order"]):
                    augmented_data = []
                    augmented_labels  = []
                    for a in args["second_order"]:
                        if a == "rotation":
                            augmented_data, augmented_labels = self.__rotateData([image_gx], [labels[ind]], args, order=2)
                        elif a == "flip":
                            augmented_data, augmented_labels = self.__flipData([image_gx], [labels[ind]], args, order=2)

                        if len(adjusted_data):
                            adjusted_data = np.concatenate((adjusted_data, augmented_data), axis=0)
                            adjusted_labels = np.concatenate((adjusted_labels, augmented_labels), axis=0)
                        else:
                            adjusted_data   = augmented_data
                            adjusted_labels = augmented_labels

                else:
                    adjusted_data.append(image_gx)
                    adjusted_labels.append(labels[ind])

        return adjusted_data, adjusted_labels

    def __flipData(self, data, labels, args, order=1):

        flipped_data = []
        flipped_labels = []

        flip = args["flip"]

        n = len(data)
        for ind in range(n):
            for f in flip:
                image_fx = np.flip(data[ind], axis=f)

                if order != 2 and  len(args["second_order"]) > 0:
                    augmented_data = []
                    augmented_labels  = []
                    for a in args["second_order"]:
                        if a == "rotation":
                            augmented_data, augmented_labels = self.__rotateData([image_fx], [labels[ind]], args, order=2)
                        elif a == "gamma":
                            augmented_data, augmented_labels = self.__adjustGamma([image_fx], [labels[ind]], args, order=2)

                        if len(flipped_data):
                            flipped_data = np.concatenate((flipped_data, augmented_data),axis=0)
                            flipped_labels = np.concatenate((flipped_labels, augmented_labels),axis=0)
                        else:
                            flipped_data = augmented_data
                            flipped_labels  = augmented_labels
                else:
                    flipped_data.append(image_fx)
                    flipped_labels.append(labels[ind])

        return flipped_data, flipped_labels


    def printLog(self):

        self.printToLog({"dataset": self.config.dataset})
        self.printToLog({"seed": self.__seed})