import os
import pathlib
from math import sqrt
from sys import maxsize
from time import time

import keras.optimizers
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Conv2D, Dense, MaxPooling2D, UpSampling2D
from keras.layers.convolutional import Conv2DTranspose
from keras.layers.core import Flatten
from keras.models import Sequential, load_model
from PIL import Image

from common import (checkQuality, checkQuality2, colorDistance,
                    convertToGrayscale, getImagePixels, getSection)

# testingDir = "testingImages"
# testingEntries = os.listdir(testingDir)

# for entry in testingEntries:
#     oP = getImagePixels("testingImages", entry, resize=(256, 256))
#     oP = oP[1:oP.shape[0] - 1, 1:oP.shape[1] - 1, :]
#     nP = getImagePixels("doublyImprovedResults", entry + "-results.png")
#     print(entry + ":", str(checkQuality2(oP, nP)))

op = getImagePixels("training", "fuji.jpg")
op = op[:, :int(op.shape[1] / 2)]
npix = getImagePixels("doublyImprovedResults", "left.png")
print(checkQuality2(op, npix))
op = getImagePixels("training", "fuji.jpg")
op = op[:, int(op.shape[1] / 2):]
npix = getImagePixels("doublyImprovedResults", "right.png")
print(checkQuality2(op, npix))
