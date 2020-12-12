import os
import pathlib

import keras
import numpy as np
import PIL
import PIL.Image
import tensorflow as tf
import tensorflow_datasets as tfds

from common import convertToGrayscale, getImagePixels

validationSplit = 0.2
trainingSize = 400
trainingSet = []
validationSet = []

trainingDir = "flower_photos"
trainingEntries = os.listdir(trainingDir)
for i in range(trainingSize):
    pixels = getImagePixels(trainingDir, trainingEntries[i])
    grayscalePixels = convertToGrayscale(pixels)
    if i > (1 - validationSplit) * trainingSize:
        validationSet =
