import os
from queue import PriorityQueue
from random import randint
from sys import maxsize
from time import time

import numpy as np
from PIL import Image

from common import (colorDistance, convertToGrayscale, getImagePixels,
                    getSection)


class Comparison:
    """
    Utility for pairing a color with an error value. 

    This is used when comparing the testing data with the training data. When a grayscale section in the testing
    data is compared with a grayscale section in the training data, the error between the two is matched with
    the representative color of the training grayscale section. 
    """

    def __init__(self, error, color):
        self.error = error
        self.color = color

    def __lt__(self, other):
        return self.error < other.error


def basicAgent(originalPixels, grayscalePixels):
    """
    Recolors the left half of a grayscale image using the image's k (5) most representative colors and recolors the right
    half the same image by comparing each 3x3 section to the left half.

    The k most representative colors are decided through a k-means clustering algorithm, detailed in the kMeans
    function below. The left half of the grayscale image is recolored using the centers and clusters returned by this 
    function. Then, every 3x3 section of the right half of the grayscale image is compared to a random sample of 3x3
    sections of the left half of the grayscale image. The representative colors of the k+1 (6) with the most similarity
    to each 3x3 section of the right half are analyzed for the majority color, and the center pixel of the section is
    recolored using this majority color (or the most similar color, if no majority color exists).

    """
    leftPixels = originalPixels[:, :int(originalPixels.shape[1] / 2)]
    leftGrayscalePixels = grayscalePixels[:, :int(originalPixels.shape[1] / 2)]

    k = 5
    representativeColors, clusters = kMeans(leftPixels, k)
    leftRecoloredPixels = [
        [[] for j in range(leftPixels.shape[1])] for i in range(leftPixels.shape[0])]
    # Recolor each member of each cluster using the corresponding representative color
    for i in range(k):
        for (r, c) in clusters[i]:
            # If edge pixel, simply recolor as black (border)
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
            # If edge pixel, simply recolor as black (border)
            if r == 0 or r == rightGrayscalePixels.shape[0] - 1 or c == 0 or c == rightGrayscalePixels.shape[1] - 1:
                rightRecoloredPixels[r][c] = np.array(
                    [0, 0, 0], dtype=np.uint8)
                continue
            startTime = time()
            rightGrayscaleSection = getSection(r, c, rightGrayscalePixels)
            mostSimilarSections = PriorityQueue()
            leftGrayscaleSectionsCenters = set()
            # Randomly select pixels from left side for the centers of sections to compare to
            while len(leftGrayscaleSectionsCenters) < randomSampleSize:
                leftGrayscaleSectionsCenters.add((randint(
                    1, leftGrayscalePixels.shape[0] - 2), randint(1, leftGrayscalePixels.shape[1] - 2)))
            itemsOnQueue = 0
            thresholdMSE = maxsize
            for sR, sC in leftGrayscaleSectionsCenters:
                leftGrayscaleSection = getSection(sR, sC, leftGrayscalePixels)
                distances = []
                # Find color distance between each pair of pixels in the two 3x3 sections
                for i in range((len(rightGrayscaleSection))):
                    distances.append(colorDistance(
                        rightGrayscaleSection[i], leftGrayscaleSection[i])**2)
                mse = sum(distances)
                representativeColor = leftRecoloredPixels[sR, sC]
                # Workaround for limiting number of items placed on queue for higher speeds
                if itemsOnQueue < grayscaleComparisons or thresholdMSE > mse:
                    mostSimilarSections.put(Comparison(
                        mse, representativeColor))
                    itemsOnQueue += 1
                    if thresholdMSE > mse:
                        thresholdMSE = mse

            # Pull the top k+1 comparisons (least error)
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

            # Iterate through top colors and find majority color
            calculatedColor = None
            for color in representedColors:
                if representedColors[color] > int(grayscaleComparisons / 2):
                    calculatedColor = np.array(color, dtype=np.uint8)
                    break
            # If no majority color, assign this pixel to have color of top comparison
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
    image.save(os.path.join("basicAgent", "results", "basic-agent-results-.png"))


def kMeans(pixels, k, distance=colorDistance):
    """
    Finds the k most representative colors in a list of pixels.

    The centers are initialized to random selections from the existing colors in the list of pixels. Then, the list of
    pixels is iterated through. Each pixel's color is compared to all of the existing centers, and the pixel's row and
    column values are stored in the cluster corresponding to the center to which this pixel's color has the least 
    distance. Then, the centers are recalculated using the average RGB values in each cluster, and the clusters are
    cleared. The comparison process then repeats.

    This algorithm repeats until there is no change in the values of the centers.

    """
    firstIteration = True
    centers = []
    clusters = [[] for i in range(k)]

    centersHaveChanged = True
    iteration = 0

    while centersHaveChanged:
        startTime = time()

        if firstIteration:
            for i in range(k):
                # Randomly inititalize the centers, ensuring that the same value is not assigned to multiple centers
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
                # Calculate the new centers based on the average RGB value of each cluster's pixels
                for (r, c) in cluster:
                    pixelRGB = pixels[r, c]
                    rgbSum[0] += pixelRGB[0]
                    rgbSum[1] += pixelRGB[1]
                    rgbSum[2] += pixelRGB[2]
                newCenter = np.array([rgbSum[0] /
                                      len(cluster), rgbSum[1] / len(cluster), rgbSum[2] / len(cluster)], dtype=np.uint8)
                # Store the changes between the old center and the new center to determine if there was any change
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
                    # Find the center with the least distance to this pixel's color, and assign the pixel to the
                    # corresponding cluster
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
