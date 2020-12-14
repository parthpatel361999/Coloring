import os
import pathlib
from math import sqrt
from sys import maxsize
from time import time

import keras.optimizers
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Conv2D, Dense, MaxPooling2D, UpSampling2D
from keras.layers.convolutional import Conv1D, Conv2DTranspose
from keras.layers.core import Flatten
from keras.models import Sequential, load_model
from keras.utils import plot_model
from PIL import Image

from common import (checkQuality2, colorDistance, convertToGrayscale,
                    getImagePixels, getSection)


def buildModel(inputShape):
    model = Sequential()
    model.add(Conv2D(32, 3, strides=1, padding="same", activation="relu", input_shape=inputShape))
    model.add(Conv2D(32, 3, strides=1, padding="same", activation="relu"))
    model.add(Conv2D(32, 3, strides=1, padding="same", activation="relu"))
    model.add(Conv2D(32, 3, strides=1, padding="same", activation="relu"))
    model.add(Conv2D(32, 3, strides=1, padding="same", activation="relu"))
    model.add(Conv2D(32, 3, strides=1, padding="same", activation="relu"))
    model.add(Conv2D(3, 3, padding="same", activation="sigmoid"))
    model.summary()
    model.compile(loss='mse', optimizer="adam", metrics=['accuracy'])
    return model


def doublyImprovedAgent(originalPixels, grayscalePixels, modelFilePath, loadModel=False):
    leftPixels = originalPixels[:, :int(originalPixels.shape[1] / 2)] / 255.0
    leftGrayscalePixels = grayscalePixels[:, :int(grayscalePixels.shape[1] / 2)] / 255.0

    model = None
    if loadModel:
        model = load_model(modelFilePath)
    else:
        model = buildModel(leftGrayscalePixels.shape)
        modelCheckpoint = ModelCheckpoint(modelFilePath, monitor='loss', mode='min', verbose=1, save_best_only=True)
        model.fit(np.array([leftGrayscalePixels]), np.array([leftPixels]),
                  epochs=2000,  verbose=1, callbacks=[modelCheckpoint])

    plot_model(model, to_file=os.path.join("doublyImprovedAgent", "model.png"), show_shapes=True, show_layer_names=True)

    leftResults = np.array(model.predict(np.array([leftGrayscalePixels])))[0] * 255.0
    image = Image.fromarray(leftResults.astype(np.uint8))
    image.save(os.path.join("doublyImprovedAgent", "results", "doubly-improved-agent-training-results.png"))

    rightGrayscalePixels = np.fliplr(convertToGrayscale(
        originalPixels[:, int(originalPixels.shape[1] / 2):], True) / 255.0)
    rightResults = np.fliplr(np.array(model.predict(np.array([rightGrayscalePixels])))[0] * 255.0)
    image = Image.fromarray(rightResults.astype(np.uint8))
    image.save(os.path.join("doublyImprovedAgent", "results", "doubly-improved-agent-testing-results.png"))

    recalculatedImageArray = [[[] for j in range(originalPixels.shape[1])]
                              for i in range(originalPixels.shape[0])]
    for r in range(len(recalculatedImageArray)):
        for c in range(leftResults.shape[1]):
            recalculatedImageArray[r][c] = leftResults[r][c]
        for c in range(rightResults.shape[1]):
            recalculatedImageArray[r][c + leftResults.shape[1]
                                      ] = rightResults[r][c]
    recalculatedImageArray = np.array(recalculatedImageArray, dtype=np.uint8)
    image = Image.fromarray(recalculatedImageArray)
    image.save(os.path.join("doublyImprovedAgent", "results", "doubly-improved-agent-overall-results.png"))


if __name__ == "__main__":
    originalPixels = getImagePixels("training", "fuji.jpg")
    doublyImprovedAgent(originalPixels,
                        convertToGrayscale(originalPixels, True),
                        os.path.join("doublyImprovedAgent", "best_model.h5"))
