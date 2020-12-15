import os
from math import sqrt

import numpy as np
from PIL import Image


def getImagePixels(directory, fileName, resize=None):
    """
    Returns the RGB values for all the pixels of an image file (after resizing the image if necessary)

    """
    filePath = os.path.join(directory, fileName)
    image = Image.open(filePath)
    if resize is not None:
        image = image.resize(resize)
    return np.asarray(image)


def convertToGrayscale(pixels, singleValue=False):
    """
    Converts the given pixels to grayscale

    """
    grayscalePixels = [[[] for j in range(pixels.shape[1])] for i in range(pixels.shape[0])]
    for i in range(pixels.shape[0]):
        for j in range(pixels.shape[1]):
            rgb = pixels[i, j]
            gray = 0.21 * rgb[0] + 0.72 * rgb[1] + 0.07 * rgb[2]
            grayscalePixels[i][j] = [gray] if singleValue else [gray, gray, gray]
    return np.array(grayscalePixels, dtype=np.uint8)


def colorDistance(color1, color2):
    """
    Calculates the distance between 2 colors 

    """
    intColor1 = np.array(color1, dtype=int)
    intColor2 = np.array(color2, dtype=int)
    return sqrt(2 * (intColor1[0] - intColor2[0])**2 + 4 * (intColor1[1] - intColor2[1])**2 + 3 * (intColor1[2] - intColor2[2])**2)


def getSection(r, c, pixels, singleValue=False):
    """
    Returns the RGB values for a 3x3 section surrounding a given pixel

    """
    neighbors = [(r - 1, c - 1), (r - 1, c), (r - 1, c + 1), (r, c - 1),
                 (r, c), (r, c + 1), (r + 1, c - 1), (r + 1, c), (r + 1, c + 1)]
    section = []
    for neighbor in neighbors:
        nR, nC = neighbor
        section.append([pixels[nR, nC, 0]] if singleValue else pixels[nR, nC])
    return np.array(section, dtype=np.uint8)


def checkQualityOfImage(originalPixels, newPixels):
    """
    Divides the given pixels into halves and compares these halves

    """
    leftOriginalPixels = originalPixels[:, :int(originalPixels.shape[1] / 2)]
    leftNewPixels = newPixels[:, :int(newPixels.shape[1] / 2)]
    rightOriginalPixels = originalPixels[:, int(originalPixels.shape[1] / 2):]
    rightNewPixels = newPixels[:, int(newPixels.shape[1] / 2):]
    return ((checkQualityOfSection(leftOriginalPixels, leftNewPixels), checkQualityOfSection(rightOriginalPixels, rightNewPixels)))


def checkQualityOfSection(originalPixels, newPixels):
    """
    Returns average color distance between pixels in originalPixels list and newPixels list

    """
    totalDistance = 0
    numPixels = 0
    for r in range(originalPixels.shape[0]):
        for c in range(originalPixels.shape[1]):
            if r == 0 or r == originalPixels.shape[0] - 1 or c == 0 or c == originalPixels.shape[1] - 1:
                continue
            totalDistance += colorDistance(originalPixels[r, c], newPixels[r, c])
            numPixels += 1
    return totalDistance/numPixels
