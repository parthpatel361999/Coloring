import os

import numpy as np
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
from keras.layers import Conv2D
from keras.models import Sequential, load_model
from keras.utils import plot_model
from PIL import Image

from common import convertToGrayscale, getImagePixels


def buildModel(inputShape):
    """
    Builds a model with 6 convolutional hidden layers, with 32 filters each. The final layer has 3 filters: 1 for R, 1 
    for G, and 1 for B.

    All of the hidden layers use a 3x3 window size and the Rectified Linear Unit activation function. The loss function
    is defined as mean square error, and the network uses the Adam optimizer.

    """
    model = Sequential()
    model.add(Conv2D(32, 3, strides=1, padding="same", activation="relu", input_shape=inputShape))
    model.add(Conv2D(32, 3, strides=1, padding="same", activation="relu"))
    model.add(Conv2D(32, 3, strides=1, padding="same", activation="relu"))
    model.add(Conv2D(32, 3, strides=1, padding="same", activation="relu"))
    model.add(Conv2D(32, 3, strides=1, padding="same", activation="relu"))
    model.add(Conv2D(32, 3, strides=1, padding="same", activation="relu"))
    model.add(Conv2D(3, 3, padding="same", activation="sigmoid"))
    model.summary()
    model.compile(loss='mse', optimizer="adam", metrics=['accuracy'])
    return model


def doublyImprovedAgent(originalPixels, grayscalePixels, modelFilePath, loadModel=False):
    """
    Uses a Keras model to recolor a grayscaled right half of an input image.

    If loadModel is True, this Agent uses a model that was previously saved in memory (specified by modelFilePath).
    Otherwise, this Agent builds a new model that training for 2000 epochs. During this training, the model that 
    produces the best training accuracy is saved. The final model is then replaced by this saved model.

    The results for training and testing are saved separately. In addition, the combined training and testing results
    are saved.

    """
    leftPixels = originalPixels[:, :int(originalPixels.shape[1] / 2)] / 255.0
    leftGrayscalePixels = grayscalePixels[:, :int(grayscalePixels.shape[1] / 2)] / 255.0

    model = None
    if loadModel:
        model = load_model(modelFilePath)
    else:
        model = buildModel(leftGrayscalePixels.shape)
        modelCheckpoint = ModelCheckpoint(modelFilePath, monitor='loss', mode='min', verbose=1, save_best_only=True)
        model.fit(np.array([leftGrayscalePixels]), np.array([leftPixels]),
                  epochs=2000,  verbose=1, callbacks=[modelCheckpoint])
        model = load_model(modelFilePath)

    plot_model(model, to_file=os.path.join("doublyImprovedAgent",
                                           "model.png"), show_shapes=True, show_layer_names=True)

    leftResults = np.array(model.predict(np.array([leftGrayscalePixels])))[0] * 255.0
    image = Image.fromarray(leftResults.astype(np.uint8))
    image.save(os.path.join("doublyImprovedAgent", "results", "doubly-improved-agent-training-results.png"))

    rightGrayscalePixels = np.fliplr(convertToGrayscale(
        originalPixels[:, int(originalPixels.shape[1] / 2):], True) / 255.0)
    rightResults = np.fliplr(np.array(model.predict(np.array([rightGrayscalePixels])))[0] * 255.0)
    image = Image.fromarray(rightResults.astype(np.uint8))
    image.save(os.path.join("doublyImprovedAgent", "results", "doubly-improved-agent-testing-results.png"))

    recalculatedImageArray = [[[] for j in range(originalPixels.shape[1])]
                              for i in range(originalPixels.shape[0])]
    for r in range(len(recalculatedImageArray)):
        for c in range(leftResults.shape[1]):
            recalculatedImageArray[r][c] = leftResults[r][c]
        for c in range(rightResults.shape[1]):
            recalculatedImageArray[r][c + leftResults.shape[1]
                                      ] = rightResults[r][c]
    recalculatedImageArray = np.array(recalculatedImageArray, dtype=np.uint8)
    image = Image.fromarray(recalculatedImageArray)
    image.save(os.path.join("doublyImprovedAgent", "results", "doubly-improved-agent-overall-results.png"))


if __name__ == "__main__":
    originalPixels = getImagePixels("training", "fuji.jpg")
    doublyImprovedAgent(originalPixels,
                        convertToGrayscale(originalPixels, True),
                        os.path.join("doublyImprovedAgent", "best_model.h5"))
