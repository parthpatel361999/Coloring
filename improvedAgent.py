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

    def __init__(self, nodeCounts, activationFunction=sigmoid, learningRate=0.1) -> None:
        self.activationFunction = activationFunction
        self.activationDerivative = self.ACTIVATION_DERIVATIVES[activationFunction.__name__]
        self.learningRate = learningRate

        np.random.seed(0)
        self.network = []
        for l in range(nodeCounts):
            layer = []
            weights = self.initalizeWeights(nodeCounts[l], self.PATCH_DIM**2 + 1 if l == 0 else nodeCounts[l - 1] + 1)
            for n in range(nodeCounts[l]):
                node = {self.WEIGHTS: weights[n]}
                layer.append(node)
            self.network.append(layer)

        outputNodeCount = 3
        outputLayer = []
        outputWeights = self.initalizeWeights(outputNodeCount, nodeCounts[-1] + 1)
        for n in range(outputNodeCount):
            outputNode = {self.WEIGHTS: outputWeights[n]}
            outputLayer.append(outputNode)
        self.network.append(outputLayer)

    def initalizeWeights(currNodes, prevNodes) -> float:
        return np.random.randn(currNodes, prevNodes) * np.sqrt(2 / prevNodes)  # He initialization

    def forwardPropagate(self, inputs) -> list:
        self.inputs = inputs  # Needed for backward propagation
        currLayerInputs = inputs
        for layer in self.network:
            nextLayerInputs = []
            for node in layer:
                output = node[self.WEIGHTS][0]  # Output starts at bias value
                for i in range(1, node[self.WEIGHTS]):
                    # Dot product between weights and current inputs
                    output += node[self.WEIGHTS][i] * currLayerInputs[i]
                node[self.OUTPUT] = self.activationFunction(output)
                nextLayerInputs.append(node[self.OUTPUT])
            currLayerInputs = nextLayerInputs  # Prep next layer's inputs
        return currLayerInputs  # Returns the last layer's outputs

    def backwardPropagate(self, expected) -> None:
        factors = []
        for l in reversed(range(len(self.network))):  # For each layer L in the network
            # Needed for all layers besides last (every such layer needs dL / dI_n for every node in the next layer)
            newFactors = []
            for n in range(len(self.network[l])):  # For each node N in L
                node = self.network[l][n]
                activationDerivative = self.activationDerivative(node[self.OUTPUT])  # AKA dO_n / dI_n
                baseGradient = activationDerivative  # Set gradient to activation function derivative of current node's output value
                if l == len(self.network) - 1:  # If L is the last layer in the network
                    lossFunctionDerivative = 2 * (node[self.OUTPUT] - expected[n])  # AKA dL / dO_n
                    baseGradient *= lossFunctionDerivative  # Multiply gradient by loss function derivative
                    # dO_n / dI_n * dL / dO_n = dL / dO_n
                    newFactors.append(activationDerivative * lossFunctionDerivative)
                else:
                    sumProduct = 0  # AKA dL / dO_n
                    for _n in range(len(self.network[l + 1])):
                        sumProduct += self.network[l + 1][_n][self.WEIGHTS][n] * factors[_n]
                    baseGradient *= sumProduct
                    newFactors.append(activationDerivative * sumProduct)  # dO_n / dI_n * dL / dO_n = dL / dO_n
                for w in range(len(node[self.WEIGHTS])):
                    gradient = baseGradient
                    if w != 0:  # Don't do anything if this weight is the bias
                        gradient *= self.network[l - 1][w -
                                                        1][self.OUTPUT] if l != 0 else self.inputs[w - 1]  # AKA dI_n / dw
                    node[self.WEIGHTS][w] -= self.learningRate * gradient  # Update weight
            factors = newFactors
