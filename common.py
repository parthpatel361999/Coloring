import os

import numpy as np
from PIL import Image


def getImagePixels(directory, fileName):
    filePath = os.path.join(directory, fileName)
    image = Image.open(filePath)
    return np.array(image, dtype=np.uint8)


def convertToGrayscale(pixels):
    grayscalePixels = np.zeros(shape=pixels.shape, dtype=np.uint8)
    for i in range(pixels.shape[0]):
        for j in range(pixels.shape[1]):
            rgb = pixels[i, j]
            gray = 0.21 * rgb[0] + 0.72 * rgb[1] + 0.07 * rgb[2]
            grayscalePixels[i, j] = [gray, gray, gray]
    return grayscalePixels
