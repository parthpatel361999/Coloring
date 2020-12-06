import numpy as np


def sigmoid(z) -> float:
    return 1.0/(1.0 + np.exp(-z))


def sigmoidDerivative(z) -> float:
    return sigmoid(z) * (1.0 - sigmoid(z))


def tanh(z) -> float:
    return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))


def tanhDerivative(z) -> float:
    return 1.0 - tanh(z)**2


def relu(z) -> float:
    return max(0.0, z)


def reluDerivative(z) -> float:
    return 1 if z > 0 else 0


class NeuralNetwork:
    ACTIVATION_DERIVATIVES = {
        "sigmoid": sigmoidDerivative,
        "tanh": tanhDerivative,
        "relu": reluDerivative
    }

    PATCH_DIM = 3

    WEIGHTS = "weights"
    OUTPUT = "output"
    ERROR = "error"

    def __init__(self, trainingData, percentValidation=0.9, numLayers=5, numFilters=10, activationFunction=sigmoid, learningRate=0.1) -> None:
        self.trainingData = trainingData[:, :int(trainingData.shape[1] * percentValidation)]
        self.validationData = trainingData[:, int(trainingData.shape[1] * percentValidation):]
        self.numLayers = numLayers
        self.numFilters = numFilters
        self.activationFunction = activationFunction
        self.activationDerivative = self.ACTIVATION_DERIVATIVES[activationFunction.__name__]
        self.learningRate = learningRate

        self.network = []
        rowsLimit = self.trainingData.shape[0] - 1
        colsLimit = self.trainingData.shape[1] - 1
        np.random.seed(0)
        for l in range(self.numLayers):
            layer = []

            for f in range(self.numFilters):
                layerFilter = []
                weights = np.random.randn(self.numFilters, self.PATCH_DIM**2 + 1) * np.sqrt(2 / (self.PATCH_DIM**2 + 1)
                                                                                            ) if l == 0 else np.random.randn(self.numFilters, self.numFilters)*np.sqrt(2 / (self.numFilters))
                for r in range(1, rowsLimit):
                    for c in range(1, colsLimit):
                        node = {}
                        node[self.WEIGHTS] = weights[weights.shape[0] * (r - 1) + (c - 1)]
                        layerFilter.append(node)

                layer.append(layerFilter)

            self.network.append(layer)
            if l != numLayers - 1:
                rowsLimit = max(rowsLimit - 2, self.PATCH_DIM)
                colsLimit = max(colsLimit - 2, self.PATCH_DIM)

        numHiddenLayerNodes = (rowsLimit - 1) * (colsLimit - 1) * self.numFilters + 1
        numFinalNodes = 3
        outputLayer = [{self.WEIGHTS: np.random.randn(
            numFinalNodes, numHiddenLayerNodes)*np.sqrt(2 / (numHiddenLayerNodes))} for i in range(numFinalNodes)]
        self.network.append(outputLayer)
