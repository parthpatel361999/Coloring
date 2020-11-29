from math import sqrt
from random import randint
from sys import maxsize
from time import time

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from common import getImagePixels


def basicAgent(originalPixels, grayscalePixels):
    leftPixels = originalPixels[:, :int(originalPixels.shape[1] / 2)]

    k = 5
    representativeColors, clusters = kMeans(leftPixels, k)
    for i in range(k):
        for (r, c) in clusters[i]:
            leftPixels[r][c] = representativeColors[i]

    image = Image.fromarray(leftPixels)
    image.save("test.jpg")

    return


def colorDistance(color1, color2):
    intColor1 = np.array(color1, dtype=int)
    intColor2 = np.array(color2, dtype=int)
    return sqrt(2 * (intColor1[0] - intColor2[0])**2 + 4 * (intColor1[1] - intColor2[1])**2 + 3 * (intColor1[2] - intColor2[2])**2)


def kMeans(pixels, k, distance=colorDistance):
    firstIteration = True
    centers = []
    clusters = [[] for i in range(k)]

    maxIterations = 10
    iteration = 0

    while iteration < maxIterations:

        startTime = time()

        if firstIteration:
            for i in range(k):
                center = pixels[randint(0, pixels.shape[0] - 1),
                                randint(0, pixels.shape[1] - 1)]
                alreadyPresent = False
                for presentCenter in centers:
                    if presentCenter.tolist() == center.tolist():
                        alreadyPresent = True
                        break
                while alreadyPresent:
                    center = pixels[randint(
                        0, pixels.shape[0] - 1), randint(0, pixels.shape[1] - 1)]
                    alreadyPresent = False
                    for presentCenter in centers:
                        if presentCenter.tolist() == center.tolist():
                            alreadyPresent = True
                            break
                centers.append(center)
            firstIteration = False
        else:
            for i in range(k):
                cluster = clusters[i]
                rgbSum = [0, 0, 0]
                for (r, c) in cluster:
                    pixelRGB = pixels[r, c]
                    rgbSum[0] += pixelRGB[0]
                    rgbSum[1] += pixelRGB[1]
                    rgbSum[2] += pixelRGB[2]
                centers[i] = np.array([rgbSum[0] /
                                       len(cluster), rgbSum[1] / len(cluster), rgbSum[2] / len(cluster)], dtype=np.uint8)
                clusters[i] = []

        for r in range(pixels.shape[0]):
            for c in range(pixels.shape[1]):
                minDistance = maxsize
                minClusterIndex = 0
                pixelRGB = pixels[r, c]
                for i in range(k):
                    pixelCenterDistance = distance(
                        centers[i], pixelRGB)
                    if pixelCenterDistance < minDistance:
                        minDistance = pixelCenterDistance
                        minClusterIndex = i
                clusters[minClusterIndex].append((r, c))

        print("Iteration", str(iteration) + ":",
              str(time() - startTime), "seconds")
        iteration += 1

    return centers, clusters


basicAgent(getImagePixels("training", "fuji.jpg"), [])
