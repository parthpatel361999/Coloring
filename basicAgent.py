from queue import PriorityQueue
from random import randint
from sys import maxsize
from time import time

import numpy as np
from PIL import Image

from common import colorDistance, convertToGrayscale, getImagePixels


class Comparison:
    def __init__(self, error, color):
        self.error = error
        self.color = color

    def __lt__(self, other):
        return self.error < other.error


def basicAgent(originalPixels, grayscalePixels):
    leftPixels = originalPixels[:, :int(originalPixels.shape[1] / 2)]
    leftGrayscalePixels = grayscalePixels[:, :int(originalPixels.shape[1] / 2)]

    k = 5
    representativeColors, clusters = kMeans(leftPixels, k)
    leftRecoloredPixels = [
        [[] for j in range(leftPixels.shape[1])] for i in range(leftPixels.shape[0])]
    for i in range(k):
        for (r, c) in clusters[i]:
            if r == 0 or r == leftPixels.shape[0] - 1 or c == 0 or c == leftPixels.shape[1] - 1:
                leftRecoloredPixels[r][c] = np.array([0, 0, 0], dtype=np.uint8)
            else:
                leftRecoloredPixels[r][c] = representativeColors[i]
    leftRecoloredPixels = np.array(leftRecoloredPixels, dtype=np.uint8)

    grayscaleComparisons = 6
    randomSampleSize = 10000
    rightGrayscalePixels = grayscalePixels[:, int(grayscalePixels.shape[1] / 2):]
    rightRecoloredPixels = [
        [[] for j in range(rightGrayscalePixels.shape[1])] for i in range(rightGrayscalePixels.shape[0])]
    for r in range(rightGrayscalePixels.shape[0]):
        for c in range(rightGrayscalePixels.shape[1]):
            if r == 0 or r == rightGrayscalePixels.shape[0] - 1 or c == 0 or c == rightGrayscalePixels.shape[1] - 1:
                rightRecoloredPixels[r][c] = np.array(
                    [0, 0, 0], dtype=np.uint8)
                continue
            startTime = time()
            rightGrayscaleSection = getSection(r, c, rightGrayscalePixels)
            mostSimilarSections = PriorityQueue()
            leftGrayscaleSectionsCenters = set()
            while len(leftGrayscaleSectionsCenters) < randomSampleSize:
                leftGrayscaleSectionsCenters.add((randint(
                    1, leftGrayscalePixels.shape[0] - 2), randint(1, leftGrayscalePixels.shape[1] - 2)))
            itemsOnQueue = 0
            thresholdMSE = maxsize
            for sR, sC in leftGrayscaleSectionsCenters:
                leftGrayscaleSection = getSection(sR, sC, leftGrayscalePixels)
                # leftGrayscaleSection = leftGrayscaleSections[s]
                distances = []
                for i in range((len(rightGrayscaleSection))):
                    distances.append(colorDistance(
                        rightGrayscaleSection[i], leftGrayscaleSection[i])**2)
                # mse = sqrt(sum(distances))
                mse = sum(distances)
                representativeColor = leftRecoloredPixels[sR, sC]
                # mostSimilarSections.put(Comparison(mse, representativeColor))
                if itemsOnQueue < grayscaleComparisons or thresholdMSE > mse:
                    mostSimilarSections.put(Comparison(
                        mse, representativeColor))
                    itemsOnQueue += 1
                    if thresholdMSE > mse:
                        thresholdMSE = mse

            topComparisons = []
            for _ in range(grayscaleComparisons):
                topComparisons.append(mostSimilarSections.get())
            representedColors = {}
            for comparison in topComparisons:
                color = tuple(comparison.color)
                if color in representedColors:
                    representedColors[color] += 1
                else:
                    representedColors[color] = 1

            calculatedColor = None
            for color in representedColors:
                if representedColors[color] > int(grayscaleComparisons / 2):
                    calculatedColor = np.array(color, dtype=np.uint8)
                    break
            if calculatedColor is None:
                calculatedColor = topComparisons[0].color
            rightRecoloredPixels[r][c] = calculatedColor
            print("Color Calculation (" + str(r) + "," + str(c) + "):",
                  time() - startTime, "seconds")

    rightRecoloredPixels = np.array(rightRecoloredPixels, dtype=np.uint8)

    recalculatedImageArray = [[[] for j in range(originalPixels.shape[1])]
                              for i in range(originalPixels.shape[0])]
    for r in range(len(recalculatedImageArray)):
        for c in range(leftRecoloredPixels.shape[1]):
            recalculatedImageArray[r][c] = leftRecoloredPixels[r][c]
        for c in range(rightRecoloredPixels.shape[1]):
            recalculatedImageArray[r][c + leftRecoloredPixels.shape[1]
                                      ] = rightRecoloredPixels[r][c]
    recalculatedImageArray = np.array(recalculatedImageArray, dtype=np.uint8)
    image = Image.fromarray(recalculatedImageArray)
    image.save("basic-agent-results.png")


def getSection(r, c, pixels):
    neighbors = [(r - 1, c - 1), (r - 1, c), (r - 1, c + 1), (r, c - 1),
                 (r, c), (r, c + 1), (r + 1, c - 1), (r + 1, c), (r + 1, c + 1)]
    section = []
    for neighbor in neighbors:
        nR, nC = neighbor
        section.append(pixels[nR, nC])
    return np.array(section, dtype=np.uint8)


def kMeans(pixels, k, distance=colorDistance):
    firstIteration = True
    centers = []
    clusters = [[] for i in range(k)]

    centersHaveChanged = True
    iteration = 0

    while centersHaveChanged:
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
            centerChanges = set()
            centersHaveChanged = False

            for i in range(k):
                cluster = clusters[i]
                rgbSum = [0, 0, 0]
                for (r, c) in cluster:
                    pixelRGB = pixels[r, c]
                    rgbSum[0] += pixelRGB[0]
                    rgbSum[1] += pixelRGB[1]
                    rgbSum[2] += pixelRGB[2]
                newCenter = np.array([rgbSum[0] /
                                      len(cluster), rgbSum[1] / len(cluster), rgbSum[2] / len(cluster)], dtype=np.uint8)
                # print(colorDistance(newCenter, centers[i]))
                centerChanges.add(colorDistance(newCenter, centers[i]))
                centers[i] = newCenter

            for change in centerChanges:
                if change != 0.0:
                    centersHaveChanged = True
                    break

        if centersHaveChanged:
            for i in range(k):
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

            print("K-Means Iteration", str(iteration) + ":",
                  str(time() - startTime), "seconds")
            iteration += 1

    return centers, clusters


if __name__ == "__main__":
    originalPixels = getImagePixels("training", "fuji.jpg")
    basicAgent(originalPixels, convertToGrayscale(originalPixels))
