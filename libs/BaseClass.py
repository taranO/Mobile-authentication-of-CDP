import os
import numpy as np

# ======================================================================================================================
class BaseClass():

    def oneHotLabel(self, labels, N):

        onehot_labels = []
        for value in labels:
            onehot_label = [0 for _ in range(N)]
            onehot_label[value] = 1
            onehot_labels.append(onehot_label)

        return np.asarray(onehot_labels).astype(np.float32)


    def normaliseDynamicRange(self, image):

        image -= np.min(image, axis=(0, 1), keepdims=True)
        image /= np.max(image, axis=(0, 1), keepdims=True)

        return image

    def reshapeData(self, data, shape):
        return data.reshape((-1, (*shape)))

    def printToLog(self, info, pref=""):
        for k, v in info.items():
            if isinstance(v, dict):
                print(pref + k + ': ')
                self.printToLog(v, pref="\t ")
            else:
                print(pref + k + ' = ' + str(v))

    def makeDir(self, dir):
        if not os.path.exists(dir):
            os.makedirs(dir)

        return dir