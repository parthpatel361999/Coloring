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


def getColors():
    colorsFile = open("colors.txt", "r")
    colorCenters = []
    for line in colorsFile:
        rgbStrings = line.strip().split(",")
        rgb = [int(string) for string in rgbStrings]
        colorCenters.append(rgb)
    colorsFile.close()
    return np.array(colorCenters, dtype=np.uint8)


def getClusters(pixels, colors):
    clusters = []
    for i in range(len(colors)):
        clusters.append(set())

    for r in range(pixels.shape[0]):
        for c in range(pixels.shape[1]):
            minDistance = maxsize
            minColorIndex = None
            pixelRGB = pixels[r, c]
            for i in range(len(colors)):
                pixelCenterDistance = colorDistance(
                    colors[i], pixelRGB)
                if pixelCenterDistance < minDistance:
                    minDistance = pixelCenterDistance
                    minColorIndex = i
            clusters[minColorIndex].add((r, c))

    return clusters


if __name__ == "__main__":
    # colors = getColors()
    trainingInputsDir = "training"
    trainingExpectedDir = "flower_photos"
    trainingEntries = os.listdir(trainingInputsDir)

    numTrainingImages = 10
    validationSplit = 0.2
    trainingInputs = []
    trainingExpected = []

    pixels = getImagePixels(trainingInputsDir, "fuji.jpg")
    leftPixels = pixels[:, :int(pixels.shape[1] / 2)]
    leftGrayscalePixels = convertToGrayscale(leftPixels, True) / 255.0

    # for r in range(1, leftGrayscalePixels.shape[0] - 1):
    #     for c in range(1, leftGrayscalePixels.shape[1] - 1):
    #         section = getSection(r, c, leftGrayscalePixels, True)
    #         trainingInputs.append(section)
    #         trainingExpected.append(leftPixels[r, c])

    # trainingInputs = np.array(trainingInputs) / 255.0
    # trainingExpected = np.array(trainingExpected) / 255.0
    # print(trainingInputs.shape)

    # model = Sequential()
    # model.add(Conv2D(32, 3, strides=1, padding="same", activation="relu", input_shape=leftGrayscalePixels.shape))
    # model.add(Conv2D(32, 3, strides=1, padding="same", activation="relu", input_shape=leftGrayscalePixels.shape))
    # model.add(Conv2D(32, 3, strides=1, padding="same", activation="relu", input_shape=leftGrayscalePixels.shape))
    # model.add(Conv2D(32, 3, strides=1, padding="same", activation="relu", input_shape=leftGrayscalePixels.shape))
    # model.add(Conv2D(32, 3, strides=1, padding="same", activation="relu", input_shape=leftGrayscalePixels.shape))
    # model.add(Conv2D(32, 3, strides=1, padding="same", activation="relu", input_shape=leftGrayscalePixels.shape))
    # model.add(Conv2D(3, 3, padding="same", activation="sigmoid"))
    # model.summary()

    # opt = keras.optimizers.SGD(learning_rate=0.1)
    # model.compile(loss='mse', optimizer="adam", metrics=['accuracy'])
    #es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=8)
    # mc = ModelCheckpoint('best_model.h5', monitor='loss', mode='min', verbose=1, save_best_only=True)
    # model.fit(np.array([leftGrayscalePixels]), np.array([leftPixels]) /255., epochs=2000,  verbose=1, callbacks=mc)
    model = load_model('best_modelFinalForReal.h5')

    plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

    leftResults = np.array(model.predict(np.array([leftGrayscalePixels])))[0] * 255.
    image = Image.fromarray(leftResults.astype(np.uint8))
    image.save(os.path.join("doublyImprovedResults", "left.png"))

    rightGrayscalePixels = np.fliplr(convertToGrayscale(pixels[:, int(pixels.shape[1] / 2):], True) / 255.)
    rightResults = np.fliplr(np.array(model.predict(np.array([rightGrayscalePixels])))[0] * 255.)
    image = Image.fromarray(rightResults.astype(np.uint8))
    image.save(os.path.join("doublyImprovedResults", "right.png"))

    recalculatedImageArray = [[[] for j in range(pixels.shape[1])]
                              for i in range(pixels.shape[0])]
    for r in range(len(recalculatedImageArray)):
        for c in range(leftResults.shape[1]):
            recalculatedImageArray[r][c] = leftResults[r][c]
        for c in range(rightResults.shape[1]):
            recalculatedImageArray[r][c + leftResults.shape[1]
                                      ] = rightResults[r][c]
    recalculatedImageArray = np.array(recalculatedImageArray, dtype=np.uint8)
    image = Image.fromarray(recalculatedImageArray)
    image.save(os.path.join("results", "doubly-improved-agent-results.png"))
    # testingInputs = []
    # for r in range(1, rightGrayscalePixels.shape[0] - 1):
    #     for c in range(1, rightGrayscalePixels.shape[1] - 1):
    #         section = getSection(r, c, rightGrayscalePixels, True)
    #         testingInputs.append(section)

    # testingInputs = np.array(testingInputs) / 255.0
    # testingResults = model.predict(testingInputs)
    # i = 0
    # rightPixels = [[[] for j in range(rightGrayscalePixels.shape[1])] for i in range(rightGrayscalePixels.shape[0])]
    # for r in range(len(rightPixels)):
    #     for c in range(len(rightPixels[0])):
    #         if r == 0 or r == len(rightPixels) - 1 or c == 0 or c == len(rightPixels[0]) - 1:
    #             rightPixels[r][c] = [0, 0, 0]
    #             continue
    #         rightPixels[r][c] = testingResults[i]
    #         i += 1
    # rightPixels = np.array(rightPixels) * 255.0

    # # plt.imshow(rightPixels)
    # # plt.savefig("r.png")

    # # print(checkQuality2(pixels[1:pixels.shape[0] - 1, int(pixels.shape[1] / 2) + 1:pixels.shape[1] - 1], rightPixels))
    # image = Image.fromarray(rightPixels.astype(np.uint8))
    # image.save(os.path.join("doublyImprovedResults", "okay.png"))

    # testingDir = "testingImages"
    # testingEntries = os.listdir(testingDir)
    # for entry in testingEntries:
    #     testingInputs = []
    #     pixels = convertToGrayscale(getImagePixels(testingDir, entry, (256, 256)))
    #     for r in range(1, pixels.shape[0] - 1):
    #         for c in range(1, pixels.shape[1] - 1):
    #             testingInputs.append(getSection(r, c, pixels, True))

    #     testingInputs = np.array(testingInputs) / 255.0
    #     testingResults = model.predict(testingInputs)
    #     recoloredImage = [[[] for j in range(254)] for i in range(254)]
    #     for r in range(len(testingResults)):
    #         result = np.array(testingResults[r]) * 255.0
    #         recoloredImage[int(r / 254)][r % 254] = result

    #     recoloredImage = np.array(recoloredImage, dtype=np.uint8)
    #     image = Image.fromarray(recoloredImage)
    #     image.save(os.path.join("doublyImprovedResults", entry+"-results.png"))

    # recoloredImage = np.array(recoloredImage, dtype=np.uint8)
    # image = Image.fromarray(recoloredImage)
    # image.save(os.path.join("doublyImprovedResults", "new-results.png"))
