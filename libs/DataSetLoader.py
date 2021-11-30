import json
import os
import math
import numpy as np

import scipy.signal
import skimage.io
from skimage.color import rgb2gray
from skimage.transform import rotate
from skimage.exposure import adjust_gamma
from sklearn.model_selection import train_test_split

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from libs.BaseClass import BaseClass


# ======================================================================================================================

class DataSetLoader(BaseClass):

    def __init__(self, config, args, type="train", is_debug_mode=False):

        self._is_debug_mode = is_debug_mode
        self.config = config
        self.type   = type if type else "test"
        self.symbol_size = config.dataset['args']["symbol_size"]

        self._seed = args.seed if "seed" in args else -1
        self._indices = []
        self.file_names = []
        self._code_name = args.code_name if "code_name" in args else "%04d.png"

        self._binary_codes_path = args.templates_path if "templates_path" in args else config.dataset["args"]["templates_path"]
        self._printed_codes_path = args.printed_path if "printed_path" in args else config.dataset["args"]["printed_path"]

        self._printed = []
        self._binary  = []


    def initDataSet(self):
        # load data
        self._loadData(self.config.dataset['args'])
        self._printed = self.reshapeData(self._printed, self.config.dataset['args']["target_size"])
        self._binary = self.reshapeData(self._binary, self.config.dataset['args']["template_target_size"])

        # create data generator
        datagen = ImageDataGenerator(samplewise_center=False, samplewise_std_normalization=False)

        # compile the data generators
        if self.type == "train":
            self.n_batches = self._printed.shape[0] // self.config.batchsize + 1
            self.datagen = datagen.flow(x=self._printed, y=self._binary,
                                        batch_size=self.config.batchsize, shuffle=True)
        else:
            self.n_batches = self._printed.shape[0]
            self.datagen = datagen.flow(x=self._printed, y=self._binary, batch_size=1, shuffle=False)

    def _loadIndices(self, path):
        with open(path) as data_file:
            list = json.load(data_file)

        return list

    def _getIndices(self, N, args):
        np.random.seed(seed=self._seed)
        indices = np.arange(1, N+1)

        # exclude the bad codes
        indices = np.setdiff1d(indices, np.asarray(args["bad_indices"]))

        if args["test_ration"] == -1:
            test_indices = indices
            train_indices = []
            val_indices = []
        else:
            train_indices, test_indices = train_test_split(indices, test_size=args["test_ration"], shuffle=True)
            train_indices, val_indices = train_test_split(train_indices, test_size=args["val_ratio"], shuffle=True)

        return train_indices, val_indices, test_indices

    def _loadData(self, args):

        list = os.listdir(self._binary_codes_path)
        list.sort()

        if self._seed >= 0:
            args["train_indices"], args["validation_indices"], args["test_indices"] = self._getIndices(len(list), args)

        if self.type == "train":
            self._printed, self._binary = self._loadImages(list, args, args["train_indices"])
            # train data augmentation
            if args["augmentation"]:
                self._augmentTrainData(args["augmentation_args"])
        elif self.type == "validation":
            self._printed, self._binary = self._loadImages(list, args, args["validation_indices"])
        elif self.type == "test":
            self._printed, self._binary = self._loadImages(list, args, args["test_indices"])
        elif self.type == "trainx": # to regenerate train codes as well, to have a full dataset
            self._printed, self._binary = self._loadImages(list, args, args["train_indices"])

    def _loadImages(self, list, args, inds):
        self._indices = inds if not isinstance(inds, str) else self._loadIndices(inds)

        N = len(self._indices)
        printed = np.zeros((N, (*args["target_size"])))
        binary  = np.zeros((N, (*args["template_target_size"])))

        i = -1
        for ind in self._indices:
            self.file_names.append(self._code_name % ind)
            i += 1
            # load data
            image_x = skimage.io.imread(self._printed_codes_path + "/" + self._code_name % ind).astype(np.float64)
            if len(image_x.shape) < len(args["target_size"]):
                image_x = image_x.reshape((image_x.shape[0], image_x.shape[1], 1))
            image_y = skimage.io.imread(self._binary_codes_path + "/" + self._code_name % ind).astype(np.float64)

            if args["synchronize_with_template"]:
                image_x, image_y = self._synchronize(image_x, image_y)

            image_x = self._centralCrop(image_x, targen_size=args["target_size"])
            image_y = self._centralCrop(image_y, targen_size=args["template_target_size"])

            printed[i] = self.normaliseDynamicRange(image_x, args)
            binary[i] = self.normaliseDynamicRange(image_y, args)

        return printed, binary

    def _centralCrop(self, image, targen_size=[330, 330, 1]):
        if image.shape[0] <= targen_size[0] and image.shape[1] <= targen_size[1]:
            return image

        height, width = image.shape[0:2]
        top_corner = self.symbol_size*math.floor((height // 2 - targen_size[0] //2) / self.symbol_size)
        left_corner = self.symbol_size*math.floor((width // 2 - targen_size[1] //2) / self.symbol_size)

        return image[top_corner:top_corner+targen_size[0], left_corner:left_corner+targen_size[1]].reshape(targen_size)

    def _synchronize(self, image_x, image_y):
        # signals normalization
        def normnormalisation(image):
            if len(image.shape) == 3:
                image = rgb2gray(image)

            image -= np.mean(image, axis=(0, 1), keepdims=True)
            image /= np.var(image, axis=(0, 1), keepdims=True)

            return image

        image_xn = normnormalisation(image_x)
        image_yn = normnormalisation(image_y)

        # correlation
        image_z = scipy.signal.fftconvolve(image_yn, image_xn[::-1, ::-1], mode='same')

        # correlation point
        mw_z = np.argmax(np.max(image_z, axis=0))
        mh_z = np.argmax(np.max(image_z, axis=1))
        # template center
        mh = image_y.shape[0] // 2
        mw = image_y.shape[1] // 2

        if mh_z != mh or mw_z != mw:
            if mw_z < mw:
                d = mw - mw_z
                image_x = image_x[:, d:]
                image_y = image_y[:, :-d]
            elif mw < mw_z:
                d = mw_z - mw
                t = math.ceil(d / self.symbol_size)

                image_x = image_x[:, t * self.symbol_size - d:-d]
                image_y = image_y[:, t * self.symbol_size:]

            if mh_z < mh:
                d = mh - mh_z
                image_x = image_x[d:, :]
                image_y = image_y[:-d, :]
            elif mh < mh_z:
                d = mh_z - mh
                t = math.ceil(d / self.symbol_size)

                image_x = image_x[t * self.symbol_size - d:-d, :]
                image_y = image_y[t * self.symbol_size:, :]

        return image_x, image_y

    def __applyAugmentations(self, augmentations, args):

        for a in augmentations:
            if a == "rotation":
                self._printed, self._binary = self._rotateData(self._printed, self._binary, args)
            elif a == "flip":
                self._printed, self._binary = self._flipData(self._printed, self._binary, args)
            elif a == "gamma":
                self._printed, self._binary = self._adjustGamma(self._printed, self._binary, args)

    def _augmentTrainData(self, args):
        self.__applyAugmentations(args["first_order"], args)
        self.__applyAugmentations(args["second_order"], args)

    def _rotateData(self, printed, binary, args):
        angels = args["rotation_angles"]

        z, h, w, c = printed.shape
        rotated_printed = np.zeros(((1+len(angels))*z, h, w, c))
        z, h, w = binary.shape
        rotated_binary  = np.zeros(((1+len(angels))*z, h, w))

        n = len(printed)
        i = -1
        for ind in range(n):
            i += 1
            rotated_printed[i] = printed[ind]
            rotated_binary[i]  = binary[ind]
            for r in angels:
                i += 1
                rotated_printed[i] = rotate(printed[ind], r, resize=False)
                rotated_binary[i] = rotate(binary[ind], r, resize=False)

        return rotated_printed, rotated_binary

    def _adjustGamma(self, printed, binary, args):

        gamma = np.arange(args["gamma"][0], args["gamma"][1]+args["gamma"][2], args["gamma"][2])

        z, h, w, c = printed.shape
        adjusted_printed = np.zeros(((1+len(gamma))*z, h, w, c))
        z, h, w = binary.shape
        adjusted_binary  = np.zeros(((1+len(gamma))*z, h, w))

        n = len(printed)
        i = -1
        for ind in range(n):
            i += 1
            adjusted_printed[i] = printed[ind]
            adjusted_binary[i]  = binary[ind]

            for g in gamma:
                i += 1
                adjusted_printed[i] = adjust_gamma(printed[ind], gamma=g)
                adjusted_binary[i] = binary[ind]

        return adjusted_printed, adjusted_binary

    def _flipData(self, printed, binary, args):

        flip = args["flip"]

        z, h, w, c = printed.shape
        flipped_printed = np.zeros(((1+len(flip))*z, h, w, c))
        z, h, w = binary.shape
        flipped_binary  = np.zeros(((1+len(flip))*z, h, w))

        n = len(printed)
        i = -1
        for ind in range(n):
            i += 1
            flipped_printed[i] = printed[ind]
            flipped_binary[i]  = binary[ind]
            for f in flip:
                i += 1
                flipped_printed[i] = np.flip(printed[ind], axis=f)
                flipped_binary[i] = np.flip(binary[ind], axis=f)

        return flipped_printed, flipped_binary

    def getData(self):
        return self._printed, self._binary

    def getIndices(self):
        return self._indices
