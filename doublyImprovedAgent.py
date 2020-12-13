import os
import pathlib
from math import sqrt

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from keras.layers import Conv2D, Dense, MaxPooling2D, UpSampling2D
from keras.layers.convolutional import Conv2DTranspose
from keras.models import Sequential
from PIL import Image

from common import convertToGrayscale, getImagePixels

validationSplit = 0.2
trainingSize = 500
trainingInputs = []
validationInputs = []
trainingExpected = []
validationExpected = []

trainingDir = "flower_photos"
trainingEntries = os.listdir(trainingDir)
for i in range(trainingSize):
    pixels = getImagePixels(trainingDir, trainingEntries[i], (256, 256))
    grayscalePixels = convertToGrayscale(pixels, True)
    if i >= (1 - validationSplit) * trainingSize:
        validationInputs.append(grayscalePixels)
        validationExpected.append(pixels)
    else:
        trainingInputs.append(grayscalePixels)
        trainingExpected.append(pixels)

trainingInputs = np.array(trainingInputs, dtype=np.uint8) / 255
trainingExpected = np.array(trainingExpected, dtype=np.uint8) / 255
validationInputs = np.array(validationInputs, dtype=np.uint8) / 255
validationExpected = np.array(validationExpected, dtype=np.uint8) / 255
# validationExpected = validationExpected[1:validationExpected.shape[0] - 1, 1:validationExpected.shape[1] - 1, :]


model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='sigmoid', kernel_initializer='he_normal', padding="same", input_shape=(
    trainingInputs.shape[1], trainingInputs.shape[2], trainingInputs.shape[3])))
# model.add(Conv2D(5, kernel_size=(3, 3), activation='relu', kernel_initializer='he_normal'))
# model.add(Conv2D(8, kernel_size=(3, 3), activation='relu', kernel_initializer='he_normal'))
# model.add(Conv2DTranspose(8, kernel_size=(3, 3), activation='relu', kernel_initializer='he_normal'))
# model.add(Conv2DTranspose(5, kernel_size=(3, 3), activation='relu', kernel_initializer='he_normal'))
# model.add(Conv2DTranspose(5, kernel_size=(3, 3), activation='relu', kernel_initializer='he_normal'))
# model.add(Conv2D(3, kernel_size=(3, 3), activation='sigmoid', padding='same'))
model.add(Dense(32, activation="sigmoid", kernel_initializer="he_normal"))
model.add(Dense(3, activation="sigmoid", kernel_initializer="he_normal"))

model.summary()
model.compile(loss='mse',
              optimizer='adam',
              metrics=['accuracy'])
model.fit(trainingInputs, trainingExpected, batch_size=40, epochs=100,
          validation_data=(validationInputs, validationExpected), verbose=1)

testInputs = []
testDir = "testingimages"
testEntries = os.listdir(testDir)
for entry in testEntries:
    testInputs.append(convertToGrayscale(getImagePixels(testDir, entry, (256, 256)), True))
testInputs = np.array(testInputs, dtype=np.uint8) / 255.

testResults = model.predict(testInputs) * 255.
for i in range(len(testResults)):
    result = np.array(testResults[i], dtype=np.uint8)
    image = Image.fromarray(result)
    image.save(os.path.join("doublyImprovedResults", "result" + str(i) + ".png"))
