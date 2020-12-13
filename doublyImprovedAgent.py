import os
import pathlib
from math import sqrt
from sys import maxsize
from time import time

import keras.optimizers
import numpy as np
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Conv2D, Dense, MaxPooling2D, UpSampling2D
from keras.layers.convolutional import Conv2DTranspose
from keras.layers.core import Flatten
from keras.models import Sequential, load_model
from PIL import Image

from common import (colorDistance, convertToGrayscale, getImagePixels,
                    getSection)

# validationSplit = 0.2
# trainingSize = 400
# trainingInputs = []
# validationInputs = []
# trainingExpected = []
# validationExpected = []

# trainingInputsDir = "grayscaleFlowers"
# trainingExpectedDir = "flower_photos"
# trainingEntries = os.listdir(trainingInputsDir)
# for i in range(trainingSize):
#     grayscalePixels = getImagePixels(trainingInputsDir, trainingEntries[i])
#     expectedPixels = getImagePixels(trainingExpectedDir, trainingEntries[i])
#     if i >= (1 - validationSplit) * trainingSize:
#         validationInputs.append(grayscalePixels)
#         validationExpected.append(expectedPixels)
#     else:
#         trainingInputs.append(grayscalePixels)
#         trainingExpected.append(expectedPixels)

# trainingInputs = np.array(trainingInputs, dtype=np.uint8) / 255
# trainingExpected = np.array(trainingExpected, dtype=np.uint8) / 255
# validationInputs = np.array(validationInputs, dtype=np.uint8) / 255
# validationExpected = np.array(validationExpected, dtype=np.uint8) / 255
# # validationExpected = validationExpected[1:validationExpected.shape[0] - 1, 1:validationExpected.shape[1] - 1, :]


# model = Sequential()
# # model.add(Dense(8, activation="relu", kernel_initializer="he_uniform", input_shape=(
# #     trainingInputs.shape[1], trainingInputs.shape[2], trainingInputs.shape[3])))
# model.add(Conv2D(8, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(
#     trainingInputs.shape[1], trainingInputs.shape[2], trainingInputs.shape[3])))
# # model.add(MaxPooling2D((2, 2), padding='same'))
# # model.add(Conv2D(8, (2, 2), activation='relu', kernel_initializer='he_uniform', padding='same'))
# # # model.add(MaxPooling2D((2, 2), padding='same'))
# # # model.add(Conv2D(8, (2, 2), activation='relu', kernel_initializer='he_uniform', padding='same'))
# # model.add(UpSampling2D((2, 2), interpolation='bilinear'))
# model.add(Conv2DTranspose(16, kernel_size=(3, 3), activation='relu', kernel_initializer='he_normal', padding="same"))
# model.add(Conv2D(3, (2, 2), activation='sigmoid', padding='same'))

# model.summary()
# opt = keras.optimizers.Adam(learning_rate=0.1)
# model.compile(loss='mse',
#               optimizer=opt,
#               metrics=['accuracy'])
# es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
# mc = ModelCheckpoint('best_model.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)
# model.fit(trainingInputs, trainingExpected, batch_size=40, epochs=50,
#           validation_data=(validationInputs, validationExpected), verbose=1, callbacks=[es, mc])
# saved_model = load_model('best_model.h5')

# testInputs = []
# testDir = "testingimages"
# testEntries = os.listdir(testDir)
# for entry in testEntries:
#     testInputs.append(convertToGrayscale(getImagePixels(testDir, entry, (256, 256))))
# testInputs = np.array(testInputs, dtype=np.uint8) / 255.

# testResults = saved_model.predict(testInputs) * 255.
# for i in range(len(testResults)):
#     result = np.array(testResults[i], dtype=np.uint8)
#     image = Image.fromarray(result)
#     image.save(os.path.join("doublyImprovedResults", "result" + str(i) + ".png"))

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
    colors = getColors()
    trainingInputsDir = "grayscaleFlowers"
    trainingExpectedDir = "flower_photos"
    trainingEntries = os.listdir(trainingInputsDir)

    numTrainingImages = 5
    validationSplit = 0.2
    trainingInputs = []
    trainingExpected = []
    for i in range(numTrainingImages):
        startTime = time()
        grayscalePixels = getImagePixels(trainingInputsDir, trainingEntries[i])
        expectedPixels = getImagePixels(trainingExpectedDir, trainingEntries[i])
        clusters = getClusters(expectedPixels, colors)
        for j in range(len(colors)):
            cluster = clusters[j]
            expectedVector = np.zeros(shape=(colors.shape[0]))
            expectedVector[j] = 1
            for (r, c) in cluster:
                if r == 0 or r == expectedPixels.shape[0] - 1 or c == 0 or c == expectedPixels.shape[1] - 1:
                    continue
                trainingInputs.append(getSection(r, c, grayscalePixels, True) / 255.0)
                trainingExpected.append(expectedVector)
        print("Image", str(i), "setup took", str(time() - startTime), "seconds.")

    trainingInputs = np.array(trainingInputs, dtype=np.uint8)
    trainingExpected = np.array(trainingExpected, dtype=np.uint8)
    print(trainingInputs.shape)

    model = Sequential()
    model.add(Dense(20, activation="relu", input_shape=(9, 1)))
    # model.add(Dense(5, activation="relu"))
    model.add(Flatten())
    model.add(Dense(colors.shape[0], activation="softmax"))
    model.summary()

    # opt = keras.optimizers.SGD(learning_rate=0.1)
    model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
    mc = ModelCheckpoint('best_model.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)
    model.fit(trainingInputs, trainingExpected, batch_size=40, epochs=15,
              validation_split=validationSplit, verbose=1, callbacks=[es, mc])
    model = load_model('best_model.h5')

    testingInputs = []
    testingDir = "testingImages"
    testingEntries = os.listdir(testingDir)
    for entry in testingEntries:
        pixels = convertToGrayscale(getImagePixels(testingDir, entry, (256, 256)))
        for r in range(1, pixels.shape[0] - 1):
            for c in range(1, pixels.shape[1] - 1):
                testingInputs.append(getSection(r, c, pixels, True) / 255.0)
        break
    testingInputs = np.array(testingInputs, dtype=np.uint8)

    testingResults = model.predict(testingInputs)
    recoloredImage = [[[] for j in range(254)] for i in range(254)]
    for r in range(len(testingResults)):
        result = np.array(testingResults[r])
        colorIndex = result.argmax()
        recoloredImage[int(r / 254)][r % 254] = colors[colorIndex]

    recoloredImage = np.array(recoloredImage, dtype=np.uint8)
    image = Image.fromarray(recoloredImage)
    image.save(os.path.join("doublyImprovedResults", "new-results-.png"))
