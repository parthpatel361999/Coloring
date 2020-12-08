from copy import deepcopy
from math import sqrt
from random import randint, seed
from sys import maxsize
from time import time

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.ndimage.filters import gaussian_filter1d
from scipy.signal import savgol_filter

from common import (colorDistance, convertToGrayscale, getImagePixels,
                    getSection)


def sigmoid(z) -> float:
    return 1.0/(1.0 + np.exp(-z)) if z >= 0 else 1.0 - (1.0 / (1.0 + np.exp(z)))


def sigmoidDerivative(z) -> float:
    return z * (1.0 - z)


def relu(z) -> float:
    return max(0.0, z)


def reluDerivative(z) -> float:
    return 1 if z > 0 else 0


class NeuralNetwork:
    ACTIVATION_DERIVATIVES = {
        "sigmoid": sigmoidDerivative,
        "relu": reluDerivative
    }

    COLOR_IMPORTANCES = [2, 4, 3]

    INPUT_SIZE = 11

    WEIGHTS = "weights"
    OUTPUT = "output"

    def __init__(self, nodeCounts, activationFunction=sigmoid, learningRate=0.03) -> None:
        self.activationFunction = activationFunction
        self.activationDerivative = self.ACTIVATION_DERIVATIVES[activationFunction.__name__]
        self.learningRate = learningRate
        self.inputs = np.array([])

        np.random.seed(0)
        self.network = []
        for l in range(len(nodeCounts)):
            layer = []
            weights = self.initalizeWeights(nodeCounts[l], self.INPUT_SIZE + 1 if l == 0 else nodeCounts[l - 1] + 1)
            for n in range(nodeCounts[l]):
                node = {self.WEIGHTS: weights[n]}
                layer.append(node)
            self.network.append(layer)

        outputNodeCount = 1
        outputLayer = []
        outputWeights = self.initalizeWeights(outputNodeCount, nodeCounts[-1] + 1)
        for n in range(outputNodeCount):
            outputNode = {self.WEIGHTS: outputWeights[n]}
            outputLayer.append(outputNode)
        self.network.append(outputLayer)

    def initalizeWeights(self, currNodes, prevNodes) -> float:
        return np.random.randn(currNodes, prevNodes) * np.sqrt(2 / prevNodes)  # He initialization

    def forwardPropagate(self, inputs, validating=False) -> list:
        if not validating:
            self.inputs = inputs  # Needed for backward propagation

        currLayerInputs = inputs
        forwardPropNetwork = deepcopy(self.network) if validating else self.network
        for l in range(len(forwardPropNetwork)):
            nextLayerInputs = []
            for node in forwardPropNetwork[l]:
                output = node[self.WEIGHTS][-1]  # Output starts at bias value
                for i in range(len(node[self.WEIGHTS]) - 1):
                    # Dot product between weights and current inputs
                    output += node[self.WEIGHTS][i] * currLayerInputs[i]
                if l == len(forwardPropNetwork) - 1:
                    # print(l, output)
                    node[self.OUTPUT] = sigmoid(output)
                else:
                    node[self.OUTPUT] = self.activationFunction(output)
                nextLayerInputs.append(node[self.OUTPUT])
            currLayerInputs = nextLayerInputs  # Prep next layer's inputs
        return currLayerInputs  # Returns the last layer's outputs

    def backwardPropagate(self, expected, lastLayerScalar=1) -> None:
        if self.inputs.size == 0:  # Need to have forward propagated first
            return

        # Needed for all layers besides last (every such layer needs dL / dI_n for every node in the next layer)
        factors = []
        for l in reversed(range(len(self.network))):  # For each layer L in the network
            newFactors = []
            for n in range(len(self.network[l])):  # For each node N in L
                node = self.network[l][n]
                baseGradient = 1  # Set gradient to activation function derivative of current node's output value
                if l == len(self.network) - 1:  # If L is the last layer in the network
                    # AKA dL / dO_n
                    lossFunctionDerivative = 2 * (node[self.OUTPUT] - expected[n]) * lastLayerScalar
                    # Multiply gradient by loss function derivative
                    baseGradient *= lossFunctionDerivative * sigmoidDerivative(node[self.OUTPUT])
                    # dO_n / dI_n * dL / dO_n = dL / dO_n
                else:
                    activationDerivative = self.activationDerivative(node[self.OUTPUT])  # AKA dO_n / dI_n
                    sumProduct = 0  # AKA dL / dO_n
                    for _n in range(len(self.network[l + 1])):
                        sumProduct += self.network[l + 1][_n][self.WEIGHTS][n] * factors[_n]
                    baseGradient *= sumProduct * activationDerivative  # dO_n / dI_n * dL / dO_n = dL / dO_n
                newFactors.append(baseGradient)
                for w in range(len(node[self.WEIGHTS])):
                    gradient = baseGradient
                    if w != len(node[self.WEIGHTS]) - 1:  # Don't do anything if this weight is the bias
                        gradient *= self.network[l - 1][w][self.OUTPUT] if l != 0 else self.inputs[w]  # AKA dI_n / dw
                    node[self.WEIGHTS][w] -= self.learningRate * gradient  # Update weight
            factors = newFactors

    def train(self, trainingInputs, trainingExpected, errorScalar=1, smoothingFactor=200, validationPercent=0.1, minEpochs=4000) -> None:
        trainingInputData = trainingInputs[:, : int(
            trainingInputs.shape[1] * (1 - validationPercent))].reshape(-1, self.INPUT_SIZE)
        validationInputData = trainingInputs[:,  int(
            trainingInputs.shape[1] * (1 - validationPercent)):].reshape(-1, self.INPUT_SIZE)
        trainingExpectedData = trainingExpected[:, :
                                                int(trainingExpected.shape[1] * (1 - validationPercent))].reshape(-1, 1)
        validationExpectedData = trainingExpected[:,
                                                  int(trainingExpected.shape[1] * (1 - validationPercent)):].reshape(-1, 1)
        trainingErrors = []
        validationErrors = []

        epoch = 0
        validationImproving = True
        minValidationErrorEpoch = epoch
        networks = []

        seed(0)
        while epoch < minEpochs or validationImproving:
            print("EPOCH", str(epoch))

            stochasticIndex = randint(0, len(trainingInputData) - 1)
            trainingOutputs = self.forwardPropagate(trainingInputData[stochasticIndex])
            self.backwardPropagate(trainingExpectedData[stochasticIndex], errorScalar)
            scaledTrainingOutputs = 255 * np.array(trainingOutputs)
            scaledTrainingExpected = 255 * np.array(trainingExpectedData[stochasticIndex])
            trainingErrors.append(sqrt(errorScalar * (scaledTrainingOutputs[0] - scaledTrainingExpected[0])**2))

            stochasticIndex = randint(0, len(validationInputData) - 1)
            validationOutputs = self.forwardPropagate(validationInputData[stochasticIndex], True)
            scaledValidationOutputs = 255 * np.array(validationOutputs)
            scaledValidationExpected = 255 * np.array(validationExpectedData[stochasticIndex])
            validationError = sqrt(errorScalar * (scaledValidationOutputs[0] - scaledValidationExpected[0])**2)
            validationErrors.append(validationError)

            networks.append(deepcopy(self.network))

            if epoch >= minEpochs:
                smoothedValidationErrors = gaussian_filter1d(validationErrors, smoothingFactor)
                optima = np.where(
                    np.diff(np.sign(np.gradient(smoothedValidationErrors))))[0]
                if len(optima) >= 1:
                    minValidationErrorEpoch = np.where(smoothedValidationErrors == min(smoothedValidationErrors))[0][0]
                    self.network = networks[minValidationErrorEpoch]
                    validationImproving = False

            epoch += 1

        return (np.array(trainingErrors), np.array(validationErrors), minValidationErrorEpoch)


def improvedAgent(originalPixels, grayscalePixels):
    leftPixels = originalPixels[:, :int(originalPixels.shape[1] / 2)]
    leftGrayscalePixels = grayscalePixels[:, :int(originalPixels.shape[1] / 2)]

    trainingInputs = []
    redTrainingExpected = []
    greenTrainingExpected = []
    blueTrainingExpected = []
    for r in range(1, leftGrayscalePixels.shape[0] - 1):
        trainingInputsRow = []
        redTrainingExpectedRow = []
        greenTrainingExpectedRow = []
        blueTrainingExpectedRow = []
        for c in range(1, leftGrayscalePixels.shape[1] - 1):
            section = np.append(getSection(r, c, leftGrayscalePixels, True), np.array([r, c]))
            trainingInputsRow.append(section)
            redTrainingExpectedRow.append([leftPixels[r, c][0]])
            greenTrainingExpectedRow.append([leftPixels[r, c][1]])
            blueTrainingExpectedRow.append([leftPixels[r, c][2]])
        trainingInputs.append(trainingInputsRow)
        redTrainingExpected.append(redTrainingExpectedRow)
        greenTrainingExpected.append(greenTrainingExpectedRow)
        blueTrainingExpected.append(blueTrainingExpectedRow)
    trainingInputs = np.array(trainingInputs) / 255.0
    redTrainingExpected = np.array(redTrainingExpected) / 255.0
    greenTrainingExpected = np.array(greenTrainingExpected) / 255.0
    blueTrainingExpected = np.array(blueTrainingExpected) / 255.0

    nodeCounts = [20, 30, 20]
    minEpochs = 50000
    smoothingFactor = int(minEpochs / 18)

    redNetwork = NeuralNetwork(nodeCounts)
    redTrainingErrors, redValidationErrors, redMinValidationError = redNetwork.train(
        trainingInputs, redTrainingExpected, errorScalar=2, minEpochs=minEpochs, smoothingFactor=smoothingFactor)
    x = np.arange(0, len(redTrainingErrors))
    redTrainingErrorsSmooth = gaussian_filter1d(redTrainingErrors, smoothingFactor)
    redValidationErrorsSmooth = gaussian_filter1d(redValidationErrors, smoothingFactor)
    figure = plt.figure(figsize=((20., 8.)))
    plt.scatter(x, redTrainingErrors, s=1, color="blue")
    plt.scatter(x, redValidationErrors, s=1, color="red")
    plt.plot(x, redTrainingErrorsSmooth, color="green", label="Training Error", linewidth=3)
    plt.plot(x, redValidationErrorsSmooth, color="gold", label="Validation Error", linewidth=3)
    plt.axvline(x=redMinValidationError, color='k', label="Validation Error Minimum")
    plt.legend(loc="best")
    plt.savefig("red-training-validation-stats.png")
    plt.close(figure)

    greenNetwork = NeuralNetwork(nodeCounts)
    greenTrainingErrors, greenValidationErrors, greenMinValidationError = greenNetwork.train(
        trainingInputs, greenTrainingExpected, errorScalar=4, minEpochs=minEpochs, smoothingFactor=smoothingFactor)
    x = np.arange(0, len(greenTrainingErrors))
    greenTrainingErrorsSmooth = gaussian_filter1d(greenTrainingErrors, smoothingFactor)
    greenValidationErrorsSmooth = gaussian_filter1d(greenValidationErrors, smoothingFactor)
    figure = plt.figure(figsize=((20., 8.)))
    plt.scatter(x, greenTrainingErrors, s=1, color="blue")
    plt.scatter(x, greenValidationErrors, s=1, color="red")
    plt.plot(x, greenTrainingErrorsSmooth, color="green", label="Training Error", linewidth=3)
    plt.plot(x, greenValidationErrorsSmooth, color="gold", label="Validation Error", linewidth=3)
    plt.axvline(x=greenMinValidationError, color='k', label="Validation Error Minimum")
    plt.legend(loc="best")
    plt.savefig("green-training-validation-stats.png")
    plt.close(figure)

    blueNetwork = NeuralNetwork(nodeCounts)
    blueTrainingErrors, blueValidationErrors, blueMinValidationError = blueNetwork.train(
        trainingInputs, blueTrainingExpected, errorScalar=3, minEpochs=minEpochs, smoothingFactor=smoothingFactor)
    x = np.arange(0, len(blueTrainingErrors))
    blueTrainingErrorsSmooth = gaussian_filter1d(blueTrainingErrors, smoothingFactor)
    figure = plt.figure(figsize=((20., 8.)))
    blueValidationErrorsSmooth = gaussian_filter1d(blueValidationErrors, smoothingFactor)
    plt.scatter(x, blueTrainingErrors, s=1, color="blue")
    plt.scatter(x, blueValidationErrors, s=1, color="red")
    plt.plot(x, blueTrainingErrorsSmooth, color="green", label="Training Error", linewidth=3)
    plt.plot(x, blueValidationErrorsSmooth, color="gold", label="Validation Error", linewidth=3)
    plt.axvline(x=blueMinValidationError, color='k', label="Validation Error Minimum")
    plt.legend(loc="best")
    plt.savefig("blue-training-validation-stats.png")
    plt.close(figure)

    leftRecoloredPixels = [[[] for j in range(leftPixels.shape[1])] for i in range(leftPixels.shape[0])]
    for r in range(0, leftGrayscalePixels.shape[0]):
        for c in range(0, leftGrayscalePixels.shape[1]):
            if r == 0 or r == leftPixels.shape[0] - 1 or c == 0 or c == leftPixels.shape[1] - 1:
                leftRecoloredPixels[r][c] = np.array([0, 0, 0], dtype=np.uint8)
                continue
            section = np.append(getSection(r, c, leftGrayscalePixels, True), np.array([r, c]))
            redOutput = 255 * np.array(redNetwork.forwardPropagate(section))
            greenOutput = 255 * np.array(greenNetwork.forwardPropagate(section))
            blueOutput = 255 * np.array(blueNetwork.forwardPropagate(section))
            # print(np.array(outputs, dtype=np.uint8))
            # print(leftPixels[r, c])
            # input()
            leftRecoloredPixels[r][c] = np.array([redOutput[0], greenOutput[0], blueOutput[0]], dtype=np.uint8)
    leftRecoloredPixels = np.array(leftRecoloredPixels, dtype=np.uint8)
    image = Image.fromarray(leftRecoloredPixels)
    image.save("improved-agent-training-results.png")


if __name__ == "__main__":
    originalPixels = getImagePixels("training", "astroworld.jpeg")
    # astroworld, starboy, after hours, bobby tarantino,
    improvedAgent(originalPixels, convertToGrayscale(originalPixels))
