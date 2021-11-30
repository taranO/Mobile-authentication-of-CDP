import numpy as np
import os
import sys

from tensorflow.keras.models import *
from tensorflow.keras.layers import *


# ======================================================================================================================
class BaseUnet():
    def __init__(self):
        super(BaseUnet, self).__init__(name='base')
        self.layer_normalisation = False

    def encoder_block(self, x, filters, kernel_size, downsample=False):
        conv_kwargs = dict(
            activation='relu',
            padding='same',
            # strides=(1,1),
            kernel_initializer='he_normal',
            data_format='channels_last'  # (batch, height, width, channels)
        )

        # Downsample input to halve Height and Width dimensions
        if downsample:
            x = MaxPooling2D(pool_size=(2, 2))(x)

        # Convolve
        x = Conv2D(filters, kernel_size, **conv_kwargs)(x)
        x = Conv2D(filters, kernel_size, **conv_kwargs)(x)
        if self.layer_normalisation:
            x = LayerNormalization(axis=(1, 2))(x)

        return x


    def decoder_block(self, inputs, filters, kernel_size, layer_normalisation=False, transpose=True, strides=2):
        x, shortcut = inputs

        conv_kwargs = dict(
            activation='relu',
            padding='same',
            kernel_initializer='he_normal',
            data_format='channels_last'  # (batch, height, width, channels)
        )

        # Upsample input to double Height and Width dimensions
        if transpose:
            # Transposed convolution a.k.a fractionally-strided convolution
            # or deconvolution although use of the latter term is confused.
            # Excellent explanation: https://github.com/vdumoulin/conv_arithmetic
            up = Conv2DTranspose(filters, 2, strides=strides, **conv_kwargs)(x)
        else:
            # Upsampling by simply repeating rows and columns then convolve
            up = UpSampling2D(size=(2, 2))(x)
            up = Conv2D(filters, 2, **conv_kwargs)(up)

        # Concatenate u-net shortcut to input
        x = concatenate([shortcut, up], axis=3)

        # Convolve
        x = Conv2D(filters, kernel_size, **conv_kwargs)(x)
        x = Conv2D(filters, kernel_size, **conv_kwargs)(x)
        if self.layer_normalisation:
            x = LayerNormalization(axis=(1, 2))(x)

        return x

    def debug_print(self, x):

        is_debug = False
        gettrace = getattr(sys, 'gettrace', None)

        if gettrace():
            is_debug = True

        if is_debug:
            print(x)

# === classical UNet ===================================================================================================
class UNet(BaseUnet):
    def __init__(self, filters=[16, 64, 128, 256, 512, 1024], layer_normalisation=False,
                 output_channels=1, transpose=True, name="UNet"):

        self.output_channels = output_channels
        self.transpose = transpose
        self.filters = filters
        self.layer_normalisation = layer_normalisation
        self.model_name = name

    # init forward pass
    def init(self, inputs):

        e1 = self.encoder_block(inputs, self.filters[1], 5, downsample=False)
        e2 = self.encoder_block(e1, self.filters[2], 3, downsample=True)
        e3 = self.encoder_block(e2, self.filters[3], 3, downsample=True)
        e4 = self.encoder_block(e3, self.filters[4], 3, downsample=True)
        e4 = Dropout(0.5)(e4)

        e5 = self.encoder_block(e4, self.filters[5], 3, downsample=True)
        e5 = Dropout(0.5)(e5)

        d6 = self.decoder_block([e5, e4], self.filters[4], 3, transpose=self.transpose)
        d7 = self.decoder_block([d6, e3], self.filters[3], 3, transpose=self.transpose)
        d8 = self.decoder_block([d7, e2], self.filters[2], 3, transpose=self.transpose)
        d9 = self.decoder_block([d8, e1], self.filters[1], 3, transpose=self.transpose)

        # Ouput
        op = Conv2D(self.filters[0], 3, padding='same', kernel_initializer='he_normal')(d9)
        op = ReLU()(op)
        op = Conv2D(self.output_channels, 1)(op)
        op = Activation('sigmoid', name='sigmoid')(op)

        return Model(inputs=[inputs], outputs=[op],  name=self.model_name)
