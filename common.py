import os
import numpy as np
from math import sqrt
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

# Returns the total color distance between two images divided by the number of pixels. (Only uses the right half)
def checkQuality(originalDirectory, originalFileName, newDirectory, newFileName):
    originalPixels = getImagePixels(originalDirectory, originalFileName)
    newPixels = getImagePixels(newDirectory, newFileName)
    rightOriginalPixels = originalPixels[:, int(originalPixels.shape[1] / 2):]
    rightNewPixels = newPixels[:, int(newPixels.shape[1] / 2):]
    totalDistance = 0
    numPixels = 0

    for r in range(rightOriginalPixels.shape[0]):
        for c in range(rightOriginalPixels.shape[1]):
            if r == 0 or r == rightOriginalPixels.shape[0] - 1 or c == 0 or c == rightOriginalPixels.shape[1] - 1:
                continue
            totalDistance += colorDistance(rightOriginalPixels[r, c], rightNewPixels[r, c])
            numPixels += 1
    return totalDistance/numPixels
