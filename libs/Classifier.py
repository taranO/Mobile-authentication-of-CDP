import numpy as np
import os
import sys

import tensorflow as tf
import keras
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.initializers import RandomNormal
import keras.backend as K

# ======================================================================================================================
class Classifier():
    def __init__(self,
                 conv_filters=[64, 32, 16, 512, 128],
                 dense_filters=[64, 32, 16, 512, 128],
                 layer_normalisation=False,
                 n_classes=1,
                 kernel_size=5,
                 stride=1,
                 max_pool=False,
                 temperature=1,
                 transpose=True, name="Classifier"):
        super(Classifier, self).__init__()

        self.n_classes = n_classes
        self._transpose = transpose
        self.conv_filters = conv_filters
        self.dense_filters = dense_filters
        self._kernel_size = kernel_size
        self._stride = stride
        self._layer_normalisation = layer_normalisation
        self._max_pool = max_pool
        self._temperature = temperature
        self.model_name = name

    # init forward pass
    def init(self, inputs):
        d = inputs
        for i, filter in enumerate(self.conv_filters):
            d = Conv2D(filter,
                       kernel_size=self._kernel_size,
                       strides=self._stride,
                       activation="relu")(d) 

            if self._layer_normalisation:
                d = LayerNormalization(axis=(1, 2))(d)

            if self._max_pool: 
                d = MaxPooling2D(pool_size=(3, 3))(d)

        d = Flatten()(d)
        for filter in self.dense_filters:
            d = Dense(units=filter, activation="relu")(d)
            d = ReLU()(d)

        d = Dense(self.n_classes)(d)
        patch_out = Softmax()(d)

        return Model(inputs=[inputs], outputs=[patch_out],  name="Classifier")

