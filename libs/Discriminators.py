import numpy as np
import os
import sys

from tensorflow.keras.models import *
from tensorflow.keras.layers import *

# ======================================================================================================================
class ClassicalDiscriminator():
    def __init__(self, filters=[64, 128, 256, 512, 512, 1], kernel_size=3, name="ClassicalDiscriminator"):

        self.filters = filters
        self.kernel_size = kernel_size
        self.__name = name

    # init forward pass
    def init(self, inputs):

        d = Conv2D(self.filters[0], self.kernel_size, strides=(2, 2), padding='same')(inputs)
        d = LeakyReLU(alpha=0.2)(d)

        d = Conv2D(self.filters[1], self.kernel_size, strides=(2, 2), padding='same')(d)
        d = BatchNormalization()(d)
        d = LeakyReLU(alpha=0.2)(d)

        d = Conv2D(self.filters[2], self.kernel_size, strides=(2, 2), padding='same')(d)
        d = BatchNormalization()(d)
        d = LeakyReLU(alpha=0.2)(d)

        d = Conv2D(self.filters[3], self.kernel_size, strides=(2, 2), padding='same')(d)
        d = BatchNormalization()(d)
        d = LeakyReLU(alpha=0.2)(d)

        d = Conv2D(self.filters[4], self.kernel_size, padding='same')(d)
        d = BatchNormalization()(d)
        d = LeakyReLU(alpha=0.2)(d)

        d = Conv2D(self.filters[5], self.kernel_size, padding='valid')(d)
        d = LeakyReLU(alpha=0.2)(d)

        d = Flatten()(d)
        d = Dense(units=1)(d)
        patch_out = Activation('sigmoid')(d)

        return Model(inputs=[inputs], outputs=[patch_out],  name=self.__name)
