import os
import sys
import json
import numpy as np
import logging as log
import matplotlib.pyplot as plt

from mpl_toolkits import axes_grid1
from skimage.color import rgb2gray
from skimage.util.shape import view_as_windows

# ======================================================================================================================
def set_log_config(is_debug=True):
    log_level = log.DEBUG if is_debug else log.INFO
    log.basicConfig(stream=sys.stdout, format='%(levelname)s: %(asctime)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S', level=log_level)

def normaliseDR(image):
    """Normalizes an input image dynamic range to [0, 1]

    Args:
        image: image to normalize

    Returns:
        np.ndarray: normalized image"""

    image -= np.min(image, axis=(0, 1), keepdims=True)
    image /= np.max(image, axis=(0, 1), keepdims=True)

    return image

def normNormalisation(image):
    """Converts an input image to the grayscale and normalizes it to be zero mean and unit norm

    Args:
        image: image to normalize

    Returns:
        np.ndarray: normalized image"""

    if len(image.shape) == 3:
        image = rgb2gray(image)

    image -= np.mean(image, axis=(0, 1), keepdims=True)
    image /= np.var(image, axis=(0, 1), keepdims=True)

    return image

def postProcessingSimbolWise(image, symbol_size=5, thr=0.5):
    """An input image binarization with guaranteed integrity of each symbol

    Args:
        image: code to process
        symbol_size: size of the symbols' blocks in an input image
        thr: binarization threshold

    Returns:
        np.ndarray: binarized code"""

    symbols = view_as_windows(image, window_shape=symbol_size, step=symbol_size)
    symbols = symbols.reshape(symbols.shape[0], symbols.shape[1], -1)
    symbols = np.mean(symbols, axis=2)
    symbols[symbols < thr] = 0
    symbols[symbols != 0] = 1

    return np.repeat(np.repeat(symbols, symbol_size, axis=0), symbol_size, axis=1)

def makeDir(dir):
    """Verifies if a directory exists and creates it if does not exist

    Args:
        dir: directory path

    Returns:
        str: directory path """

    if not os.path.exists(dir):
        os.makedirs(dir)

    return dir

def oneHotLabel(labels):
    """Performs an encoding of the decimal labels to the one-hot vectors representation

    Args:
        labels: a vector of the decimal labels

    Returns:
        np.ndarray: the one-hot vectors representation """

    onehot_labels = []
    for value in labels:
        onehot_label = [0 for _ in range(10)]
        onehot_label[value] = 1
        onehot_labels.append(onehot_label)

    return np.asarray(onehot_labels).astype(np.float32)


def add_colorbar(im, aspect=20, pad_fraction=0.5, **kwargs):
    """Add a vertical color bar of the size of an image plot """

    divider = axes_grid1.make_axes_locatable(im.axes)
    width = axes_grid1.axes_size.AxesY(im.axes, aspect=1./aspect)
    pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
    current_ax = plt.gca()
    cax = divider.append_axes("right", size=width, pad=pad)
    plt.sca(current_ax)

    return im.axes.figure.colorbar(im, cax=cax, **kwargs)

def mse1D(img1, img2):
    """Measures the MSE error between the 1D signals """
    return np.sum(np.power((img1 - img2), 2)) / img1.size

def saveListAsJson(list, to_file):
    """Saves a list of data to json format"""
    with open(to_file, 'w') as f:
        f.write(json.dumps(list))

def loadListFromJson(from_file):
    """Loads the data from json to list"""
    with open(from_file, 'r') as f:
        list = json.loads(f.read())

    return list

def saveSpeed(epoch):
    """Saving speed during the model training"""
    save_each = 1
    if epoch <= 10:
        save_each = 1
    elif epoch <= 100:
        save_each = 10
    elif epoch <= 1000:
        save_each = 50
    elif epoch <= 10000:
        save_each = 100

    return save_each


