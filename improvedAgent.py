import os
from copy import deepcopy
from random import randint
from time import time

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.ndimage.filters import gaussian_filter1d

from common import (checkQualityOfImage, colorDistance, convertToGrayscale,
                    getImagePixels, getSection)


def sigmoid(z) -> float:
    return 1.0/(1.0 + np.exp(-z))


def sigmoidDerivative(z) -> float:
    return z * (1.0 - z)


def relu(z) -> float:
    return max(0.0, z)


def reluDerivative(z) -> float:
    return 1 if z > 0 else 0


class NeuralNetwork:
    """
    Custom dense neural network implementation.
    """
    ACTIVATION_DERIVATIVES = {
        "sigmoid": sigmoidDerivative,
        "relu": reluDerivative
    }

    INPUT_SIZE = 15
    OUTPUT_SIZE = 3

    WEIGHTS = "weights"
    OUTPUT = "output"

    def __init__(self, nodeCounts, activationFunction=sigmoid, learningRate=0.1) -> None:
        """
        Initializes the network with the given arguments and creates the hidden layers based on nodeCounts, a list
        containing the number of nodes per hidden layer. Also creates the output layer.

        """
        self.activationFunction = activationFunction
        self.activationDerivative = self.ACTIVATION_DERIVATIVES[activationFunction.__name__]
        self.learningRate = learningRate
        self.inputs = np.array([])

        # Create hidden layers as per nodeCounts argument, assigning random weights to each node
        self.network = []
        for l in range(len(nodeCounts)):
            layer = []
            weights = self.initalizeWeights(nodeCounts[l], self.INPUT_SIZE + 1 if l == 0 else nodeCounts[l - 1] + 1)
            for n in range(nodeCounts[l]):
                node = {self.WEIGHTS: weights[n]}
                layer.append(node)
            self.network.append(layer)

        # Create output layer with predetermined number of nodes, each with random weight assigned to it
        outputLayer = []
        outputWeights = self.initalizeWeights(self.OUTPUT_SIZE, nodeCounts[-1] + 1)
        for n in range(self.OUTPUT_SIZE):
            outputNode = {self.WEIGHTS: outputWeights[n]}
            outputLayer.append(outputNode)
        self.network.append(outputLayer)

    def initalizeWeights(self, currNodes, prevNodes) -> float:
        """
        Creates a list of weights that follow the He weight initalization algorithm.

        """
        return np.random.randn(currNodes, prevNodes) * np.sqrt(2 / prevNodes)

    def forwardPropagate(self, inputs, noChange=False) -> list:
        """
        Pushes a given input vector through the network and updates the outputs of each node if argument noChange is
        False.

        """
        if not noChange:
            # Update inputs for backward propagation
            self.inputs = inputs

        currLayerInputs = inputs
        forwardPropNetwork = deepcopy(self.network) if noChange else self.network
        for l in range(len(forwardPropNetwork)):
            nextLayerInputs = []
            for node in forwardPropNetwork[l]:
                # Initialize output as bias value
                output = node[self.WEIGHTS][-1]
                # Compute dot product between weights and current layer inputs
                for i in range(len(node[self.WEIGHTS]) - 1):
                    output += node[self.WEIGHTS][i] * currLayerInputs[i]
                # If iterating on last layer, always use sigmoid activation function
                if l == len(forwardPropNetwork) - 1:
                    node[self.OUTPUT] = sigmoid(output)
                else:
                    node[self.OUTPUT] = self.activationFunction(output)
                nextLayerInputs.append(node[self.OUTPUT])
            # Prep next layer's inputs
            currLayerInputs = nextLayerInputs
        # Returns the last layer's outputs
        return currLayerInputs

    def backwardPropagate(self, expected) -> None:
        """
        Adjusts the weights for each node based on the difference of the expected output and the output calculated by
        the network. 

        """
        if self.inputs.size == 0:
            # Need to have forward propagated first
            return

        # Maintain a list of the next layer's dL / dO_n (dynamic programming approach for gradient calculation)
        factors = []
        for l in reversed(range(len(self.network))):  # For each layer L in the network
            newFactors = []
            for n in range(len(self.network[l])):  # For each node N in L
                node = self.network[l][n]
                baseGradient = 1  # Set gradient to activation function derivative of current node's output value
                if l == len(self.network) - 1:  # If L is the last layer in the network
                    lossFunctionDerivative = 2 * (node[self.OUTPUT] - expected[n])  # dL / dO_n
                    activationDerivative = sigmoidDerivative(node[self.OUTPUT])  # dO_n / dI_n
                    baseGradient *= lossFunctionDerivative * activationDerivative  # dO_n / dI_n * dL / dO_n = dL / dI_n
                else:
                    activationDerivative = self.activationDerivative(node[self.OUTPUT])  # dO_n / dI_n
                    sumProduct = 0  # dL / dO_n
                    for _n in range(len(self.network[l + 1])):
                        sumProduct += self.network[l + 1][_n][self.WEIGHTS][n] * factors[_n]
                    baseGradient *= sumProduct * activationDerivative  # dO_n / dI_n * dL / dO_n = dL / dI_n
                newFactors.append(baseGradient)
                for w in range(len(node[self.WEIGHTS])):
                    gradient = baseGradient
                    if w != len(node[self.WEIGHTS]) - 1:  # Don't do anything if this weight is the bias
                        gradient *= self.network[l - 1][w][self.OUTPUT] if l != 0 else self.inputs[w]  # dI_n / dw
                    node[self.WEIGHTS][w] -= self.learningRate * gradient  # Update weight
            factors = newFactors

    def train(self, trainingData, validationData, smoothingFactor=200, minEpochs=4000) -> None:
        """
        Trains the network for at least the given number of epochs and saves the "best" network in terms of validation
        error.

        The state of the network is saved after every epoch, and after the network is trained for the given number of 
        epochs at minimum, the smoothingFactor argument is used to smooth the validation errors recorded at that
        point. If there is at least one optimum in the smoothed validation errors, the epoch at which the minimum
        validation error occurred is found, and the network is reverted to the saved network after that epoch. If this
        condition is not met, the network continues training until it is.

        """
        trainingInputs = trainingData[0].reshape(-1, self.INPUT_SIZE)
        trainingExpected = trainingData[1].reshape(-1, self.OUTPUT_SIZE)
        validationInputs = validationData[0].reshape(-1, self.INPUT_SIZE)
        validationExpected = validationData[1].reshape(-1, self.OUTPUT_SIZE)
        trainingErrors = []
        validationErrors = []

        epoch = 0
        validationImproving = True
        minValidationErrorEpoch = epoch
        networks = []

        prevTrainingIndices = set()
        prevValidationIndices = set()

        while epoch < minEpochs or validationImproving:
            # Randomly choose a data point to train, but ensure it hasn't been trained before
            stochasticIndex = randint(0, len(trainingInputs) - 1)
            while stochasticIndex in prevTrainingIndices:
                stochasticIndex = randint(0, len(trainingInputs) - 1)
            prevTrainingIndices.add(stochasticIndex)
            # If all data points have been trained, flush the set holding previously trained data points
            if len(prevTrainingIndices) >= len(trainingInputs):
                prevTrainingIndices = set()
            trainingOutputs = self.forwardPropagate(trainingInputs[stochasticIndex])
            self.backwardPropagate(trainingExpected[stochasticIndex])
            # Upscale the network outputs and expected outputs by 255 to determine the color distance between them
            scaledTrainingOutputs = 255 * np.array(trainingOutputs)
            scaledTrainingExpected = 255 * np.array(trainingExpected[stochasticIndex])
            trainingErrors.append(colorDistance(scaledTrainingOutputs, scaledTrainingExpected))

            # Repeat the training process on the validation data, but do not backward propagate
            stochasticIndex = randint(0, len(validationInputs) - 1)
            while stochasticIndex in prevValidationIndices:
                stochasticIndex = randint(0, len(validationInputs) - 1)
            prevValidationIndices.add(stochasticIndex)
            if len(prevValidationIndices) >= len(validationInputs):
                prevValidationIndices = set()
            validationOutputs = self.forwardPropagate(validationInputs[stochasticIndex], True)
            scaledValidationOutputs = 255 * np.array(validationOutputs)
            scaledValidationExpected = 255 * np.array(validationExpected[stochasticIndex])
            validationError = colorDistance(scaledValidationOutputs, scaledValidationExpected)
            validationErrors.append(validationError)

            # Save the network after this epoch
            networks.append(deepcopy(self.network))

            if epoch >= minEpochs:
                smoothedValidationErrors = gaussian_filter1d(validationErrors, smoothingFactor)
                optima = np.where(np.diff(np.sign(np.gradient(smoothedValidationErrors))))[0]
                if len(optima) >= 1:
                    # Revert the network to the epoch at which the minimum validation error was encountered
                    minValidationErrorEpoch = np.where(smoothedValidationErrors == min(smoothedValidationErrors))[0][0]
                    self.network = networks[minValidationErrorEpoch]
                    validationImproving = False

            epoch += 1

        return (np.array(trainingErrors), np.array(validationErrors), minValidationErrorEpoch)


def prepareInput(r, c, pixels, numSections=(15, 3)):
    """
    Creates an input vector for use in the neural network.

    The input vector contains 15 different pieces of information. The first 9 are the gray values of the 3x3 section
    surrounding the pixel at r, c in the argument list pixels. The next 2 represent the area of the dataset in which
    the pixel resides -- there are 45 segments by default (rows are separated into 15 segments, columns are separated
    into 3 segments). The following 2 represent where in the respective segment the pixel is located. The final 2
    represent where in the overall dataset the pixel is located.

    All values are normalized between 0 and 1.

    """
    rowSections, colSections = numSections
    rowDividers = np.linspace(0, pixels.shape[0], rowSections)
    colDividers = np.linspace(0, pixels.shape[1], colSections)
    patch = getSection(r, c, pixels) / 255.0
    extraInputAttributes = []
    rowIndex = 0
    for i in range(len(rowDividers)):
        if rowDividers[i] > r:
            rowIndex = i
            break
    colIndex = 0
    for i in range(len(colDividers)):
        if colDividers[i] > c:
            colIndex = i
            break
    extraInputAttributes.append(rowIndex/rowSections)
    extraInputAttributes.append(colIndex/colSections)
    extraInputAttributes.append(
        (r % int(pixels.shape[0] / rowSections)) / int(pixels.shape[0] / rowSections))
    extraInputAttributes.append(
        (c % int(pixels.shape[1] / colSections)) / int(pixels.shape[1] / colSections))
    extraInputAttributes.append(r / pixels.shape[0])
    extraInputAttributes.append(c / pixels.shape[1])
    return np.append(patch, extraInputAttributes)


def improvedAgent(originalPixels, grayscalePixels, iteration):
    """
    Uses the custom neural network to recolor a grayscale image by training on the left half of the image and testing
    on the right half of the image.

    The training data is split into actual training data and validation data based on the validationPortion variable.
    The network is then trained using this training data and validation data. The finalized model is used to recolor
    the entire training data and afterwards the entire testing data. The two resulting pixel lists are combined to
    generate an entirely recalculated image.     

    """
    leftPixels = originalPixels[:, :int(originalPixels.shape[1] / 2)]
    leftGrayscalePixels = grayscalePixels[:, :int(originalPixels.shape[1] / 2)]

    overallStartTime = time()
    startTime = time()
    validationPortion = 0.1

    # Generate the training inputs and training expected data
    trainingInputPixels = leftGrayscalePixels[:, :int(leftGrayscalePixels.shape[1] * (1 - validationPortion))]
    trainingExpectedPixels = leftPixels[:, :int(leftPixels.shape[1] * (1 - validationPortion))]
    trainingInputs = []
    trainingExpected = []
    for r in range(1, trainingInputPixels.shape[0] - 1):
        trainingInputsRow = []
        trainingExpectedRow = []
        for c in range(1, trainingInputPixels.shape[1] - 1):
            trainingInput = prepareInput(r, c, trainingInputPixels)
            trainingInputsRow.append(trainingInput)
            trainingExpectedRow.append(trainingExpectedPixels[r, c] / 255.0)
        trainingInputs.append(trainingInputsRow)
        trainingExpected.append(trainingExpectedRow)
    trainingInputs = np.array(trainingInputs)
    trainingExpected = np.array(trainingExpected)

    # Generate the validation inputs and validation expected data
    validationInputPixels = leftGrayscalePixels[:, int(leftGrayscalePixels.shape[1] * (1 - validationPortion)):]
    validationExpectedPixels = leftPixels[:, int(leftPixels.shape[1] * (1 - validationPortion)):]
    validationInputs = []
    validationExpected = []
    for r in range(1, validationInputPixels.shape[0] - 1):
        validationInputsRow = []
        validationExpectedRow = []
        for c in range(1, validationInputPixels.shape[1] - 1):
            validationInput = prepareInput(r, c, validationInputPixels)
            validationInputsRow.append(validationInput)
            validationExpectedRow.append(validationExpectedPixels[r, c] / 255.0)
        validationInputs.append(validationInputsRow)
        validationExpected.append(validationExpectedRow)
    validationInputs = np.array(validationInputs)
    validationExpected = np.array(validationExpected)

    # Initialize the network
    nodeCounts = [10]
    minEpochs = 5000
    smoothingFactor = int(minEpochs / 20)
    network = NeuralNetwork(nodeCounts, learningRate=0.4)
    print("Training data and network setup took", str(time() - startTime), "seconds.")

    # Train the network
    startTime = time()
    trainingErrors, validationErrors, minValidationError = network.train(
        (trainingInputs, trainingExpected), (validationInputs, validationExpected),
        minEpochs=minEpochs, smoothingFactor=smoothingFactor)
    print("Network training took", str(time() - startTime), "seconds.")

    # Plot the training errors and validation errors
    x = np.arange(0, len(trainingErrors))
    trainingErrorsSmooth = gaussian_filter1d(trainingErrors, smoothingFactor)
    validationErrorsSmooth = gaussian_filter1d(validationErrors, smoothingFactor)
    figure = plt.figure(figsize=((20., 8.)))
    plt.scatter(x, trainingErrors, s=1, color="blue", label="Training Error")
    plt.scatter(x, validationErrors, s=1, color="red", label="Validation Error")
    plt.plot(x, trainingErrorsSmooth, color="green", label="Training Error (Smoothed)", linewidth=3)
    plt.plot(x, validationErrorsSmooth, color="gold", label="Validation Error (Smoothed)", linewidth=3)
    plt.axvline(x=minValidationError, color='k', label="Validation Error Minimum (Smoothed)")
    plt.legend(loc="best")
    plt.savefig(os.path.join("improvedAgent", "training", "training-validation-stats-" + str(iteration) + ".png"))
    plt.close(figure)

    # Recolor left side of grayscale image
    startTime = time()
    leftRecoloredPixels = [[[] for j in range(leftPixels.shape[1])] for i in range(leftPixels.shape[0])]
    for r in range(0, leftGrayscalePixels.shape[0]):
        for c in range(0, leftGrayscalePixels.shape[1]):
            if r == 0 or r == leftGrayscalePixels.shape[0] - 1 or c == 0 or c == leftGrayscalePixels.shape[1] - 1:
                leftRecoloredPixels[r][c] = np.array([0, 0, 0], dtype=np.uint8)
                continue
            trainInput = prepareInput(r, c, leftGrayscalePixels)
            networkOutputs = network.forwardPropagate(trainInput, True)
            output = 255 * np.array(networkOutputs)
            leftRecoloredPixels[r][c] = np.array(output, dtype=np.uint8)
    leftRecoloredPixels = np.array(leftRecoloredPixels, dtype=np.uint8)
    print("Calculating left side of image took", str(time() - startTime), "seconds.")

    # Recolor the right side of grayscale image, but flip the columns to bolster symmetry with left side
    rightGrayscalePixels = np.fliplr(grayscalePixels[:, int(grayscalePixels.shape[1] / 2):])
    rightRecoloredPixels = [[[] for j in range(rightGrayscalePixels.shape[1])]
                            for i in range(rightGrayscalePixels.shape[0])]
    startTime = time()
    for r in range(0, rightGrayscalePixels.shape[0]):
        for c in range(0, rightGrayscalePixels.shape[1]):
            if r == 0 or r == rightGrayscalePixels.shape[0] - 1 or c == 0 or c == rightGrayscalePixels.shape[1] - 1:
                rightRecoloredPixels[r][c] = np.array([0, 0, 0], dtype=np.uint8)
                continue
            testInput = prepareInput(r, c, rightGrayscalePixels)
            networkOutputs = network.forwardPropagate(testInput, True)
            output = 255 * np.array(networkOutputs)
            rightRecoloredPixels[r][c] = np.array(output, dtype=np.uint8)
    rightRecoloredPixels = np.fliplr(np.array(rightRecoloredPixels, dtype=np.uint8))
    print("Predicting right side of image took", str(time() - startTime), "seconds.")

    # Combine recolored left half and right half
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
    image.save(os.path.join("improvedAgent", "results", "improved-agent-results-" + str(iteration) + ".png"))
    print("Overall took", str(time() - overallStartTime), "seconds.")
    return time() - overallStartTime


if __name__ == "__main__":
    originalPixels = getImagePixels("training", "fuji.jpg")
    iterations = 1
    overallTrainingError = 0
    overallTestingError = 0
    overallTime = 0
    for i in range(iterations):
        print("ITERATION", str(i))
        overallTime += improvedAgent(originalPixels, convertToGrayscale(originalPixels, True), i)
        newPixels = getImagePixels("results", "improved-agent-results-" + str(i) + ".png")
        trainingError, testingError = checkQualityOfImage(originalPixels, newPixels)
        overallTrainingError += trainingError
        overallTestingError += testingError
    print("TRAINING ERROR:", str(overallTrainingError / iterations))
    print("TESTING ERROR:", str(overallTestingError / iterations))
    print("RUNTIME:", str(overallTime / iterations))
