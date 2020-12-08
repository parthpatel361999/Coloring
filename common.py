import os
from math import sqrt

import numpy as np
from PIL import Image


def getImagePixels(directory, fileName):
    filePath = os.path.join(directory, fileName)
    image = Image.open(filePath)
    return np.asarray(image)


def convertToGrayscale(pixels):
    grayscalePixels = np.zeros(shape=pixels.shape)
    for i in range(pixels.shape[0]):
        for j in range(pixels.shape[1]):
            rgb = pixels[i, j]
            gray = 0.21 * rgb[0] + 0.72 * rgb[1] + 0.07 * rgb[2]
            grayscalePixels[i, j] = [gray, gray, gray]
    return grayscalePixels


def colorDistance(color1, color2):
    intColor1 = np.array(color1, dtype=int)
    intColor2 = np.array(color2, dtype=int)
    return sqrt(2 * (intColor1[0] - intColor2[0])**2 + 4 * (intColor1[1] - intColor2[1])**2 + 3 * (intColor1[2] - intColor2[2])**2)


def getSection(r, c, pixels, grayscale=False):
    neighbors = [(r - 1, c - 1), (r - 1, c), (r - 1, c + 1), (r, c - 1),
                 (r, c), (r, c + 1), (r + 1, c - 1), (r + 1, c), (r + 1, c + 1)]
    section = []
    for neighbor in neighbors:
        nR, nC = neighbor
        section.append(pixels[nR, nC][0] if grayscale else pixels[nR, nC])
    return np.array(section, dtype=np.uint8)
