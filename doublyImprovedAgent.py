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
from PIL import Image

from common import (colorDistance, convertToGrayscale, getImagePixels,
                    getSection)


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
    leftGrayscalePixels = convertToGrayscale(leftPixels, True)

    for r in range(1, leftGrayscalePixels.shape[0] - 1):
        for c in range(1, leftGrayscalePixels.shape[1] - 1):
            section = getSection(r, c, leftGrayscalePixels, True)
            trainingInputs.append(section)
            trainingExpected.append(leftPixels[r, c])

    trainingInputs = np.array(trainingInputs) / 255.0
    trainingExpected = np.array(trainingExpected) / 255.0
    print(trainingInputs.shape)

    model = Sequential()
    model.add(Dense(20, activation="relu", input_shape=(9, 1)))
    model.add(Dense(20, activation="relu"))
    model.add(Dense(20, activation="relu"))
    # model.add(Conv1D(5, 3, strides=1, activation="relu"))
    # model.add(Dense(10, activation="relu"))
    # model.add(Dense(10, activation="relu"))
    # model.add(Dense(10, activation="relu"))
    model.add(Flatten())
    model.add(Dense(3, activation="sigmoid"))
    model.summary()

    # opt = keras.optimizers.SGD(learning_rate=0.1)
    model.compile(loss='mse', optimizer="sgd", metrics=['accuracy'])
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=8)
    mc = ModelCheckpoint('best_model.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)
    model.fit(trainingInputs, trainingExpected, batch_size=25, epochs=100,
              validation_split=validationSplit, verbose=1, callbacks=[mc, es])
    model = load_model('best_model-2.h5')

    rightGrayscalePixels = convertToGrayscale(pixels[:, int(pixels.shape[1] / 2):])
    testingInputs = []
    for r in range(1, rightGrayscalePixels.shape[0] - 1):
        for c in range(1, rightGrayscalePixels.shape[1] - 1):
            section = getSection(r, c, rightGrayscalePixels, True)
            testingInputs.append(section)

    testingInputs = np.array(testingInputs) / 255.0
    testingResults = model.predict(testingInputs)
    i = 0
    rightPixels = [[[] for j in range(rightGrayscalePixels.shape[1])] for i in range(rightGrayscalePixels.shape[0])]
    for r in range(len(rightPixels)):
        for c in range(len(rightPixels[0])):
            if r == 0 or r == len(rightPixels) - 1 or c == 0 or c == len(rightPixels[0]) - 1:
                rightPixels[r][c] = [0, 0, 0]
                continue
            rightPixels[r][c] = testingResults[i]
            i += 1
    rightPixels = np.array(rightPixels)
    plt.imshow(rightPixels)
    plt.savefig("r.png")

    # --------------------------------
    # image = Image.fromarray(rightPixels)
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
