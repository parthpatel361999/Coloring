import os
from random import choices, randint
from string import ascii_uppercase

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential, load_model
from PIL import Image, ImageDraw, ImageFont
from skimage import color, io
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential


def modelbuilder():
    """
    Builds a sequential TensorFlow model

    """
    model = Sequential()
    # Add a convolution layer with with a sigmoid activation function
    model.add(layers.Conv2D(1, (2, 2), strides=(1, 1), activation='sigmoid', padding='same', input_shape=(256, 256, 3)))
    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    model.summary()
    return model


def drawText(filepath, fontsize, text, target_size, color):
    """
    Draws customized text on images and track the position of the text

    """
    image = Image.open(filepath)
    image = image.resize(target_size)
    draw = ImageDraw.Draw(image)
    # Import the arial font style
    font = ImageFont.truetype(font=os.path.join("inpainting", "arial.ttf"), size=fontsize)
    size = font.getsize(text)
    x = randint(size[0] + 16, 255 - 16 - size[0])
    y = randint(size[1] + 16, 255 - 16 - size[1])
    draw.text((x, y), text, fill=color, font=font)
    # Determine the bounds of the text area
    x0 = x + 0.5 * size[0] - 16
    y1 = y + 0.5 * fontsize + 16
    x1 = x + 0.5 * size[0] + 16
    y0 = y + 0.5 * fontsize - 16
    return image, (int(x0), int(y0), int(x1), int(y1))


def getNeighbors(r, c, buffer):
    """
    Returns the neighboring pixels of a given pixel with a buffer size

    """
    neighbors = [(r - buffer, c - buffer), (r - buffer, c), (r - buffer, c + buffer), (r, c - buffer),
                 (r, c + buffer), (r + buffer, c - buffer), (r + buffer, c), (r + buffer, c + buffer)]
    section = []
    # Check if the neighbors are within the image boundaries
    for neighbor in neighbors:
        nR, nC = neighbor
        if nR >= 0 and nR < 256 and nC >= 0 and nC < 256:
            section.append([nR, nC])
    return section


def smooth(rgb_with_text, gray_detected_text):
    """
    Blends the text on an image with its background

    """
    detectedPixels = []

    # Compile a list of pixels that need to be blended
    for r in range(256):
        for c in range(256):
            if gray_detected_text[r, c] > 0.92:
                detectedPixels.append([r, c])
    length = len(detectedPixels)

    # Add neighbors of the pixels to be blended to the list
    for i in range(length):
        neighbors = getNeighbors(detectedPixels[i][0], detectedPixels[i][1], 1)
        for neighbor in neighbors:
            if neighbor not in detectedPixels:
                detectedPixels.append(neighbor)

    # Blend the text pixel's rgb values with their neighbors
    for loc in detectedPixels:
        neighbors = getNeighbors(loc[0], loc[1], 3)
        averagedPixel = [0, 0, 0]
        numPixels = 0
        for neighbor in neighbors:
            # Only average in neighboring pixels are not marked as part of the text pixel list
            if neighbor not in detectedPixels:
                averagedPixel[0] += rgb_with_text[neighbor[0]][neighbor[1]][0]
                averagedPixel[1] += rgb_with_text[neighbor[0]][neighbor[1]][1]
                averagedPixel[2] += rgb_with_text[neighbor[0]][neighbor[1]][2]
                numPixels += 1
        if numPixels > 0:
            averagedPixel[0] /= numPixels
            averagedPixel[1] /= numPixels
            averagedPixel[2] /= numPixels
            rgb_with_text[loc[0]][loc[1]] = averagedPixel
    return rgb_with_text


def inpainting(modelFilePath=os.path.join("inpainting", 'test_model_S500E500.h5'), loadModel=False):
    # Initialize variables for training and testing
    target_size = (256, 256)
    textImages = []
    expOutM1 = []
    testTextImages = []
    size = 300
    fontsize = 30

    # Retrieve and process the images for the training and validation set
    i = 0
    for filename in os.listdir(os.path.join("inpainting", "flower_photos")):
        # Radnomly generate a 4 letter word to be overlaid on an image
        message = ''.join(choices(ascii_uppercase, k=4))
        # Randomly select a color for the font
        c = 'rgb(' + str(randint(0, 255)) + ',' + str(randint(0, 255)) + ',' + str(randint(0, 255)) + ')'
        # Draw the words on the images
        images = drawText(os.path.join("inpainting", "flower_photos",
                                       filename), fontsize, message, target_size, c)
        textImages.append(np.array(images[0], dtype=float))
        # Keep track of the location of the word for each image
        temp = np.zeros(shape=(256, 256, 3), dtype=float)
        x0 = images[1][0]
        y0 = images[1][1]
        x1 = images[1][2]
        y1 = images[1][3]
        temp[y0:y1, x0:x1, :] = 1.0
        expOutM1.append(temp)
        i += 1
        if(i == size):
            break

    # Retrieve and process the images for the testing set
    i = 0
    for filename in os.listdir(os.path.join("inpainting", "testingImages")):
        # Radnomly generate a 4 letter word to be overlaid on an image
        message = ''.join(choices(ascii_uppercase, k=4))
        images = drawText(os.path.join("inpainting", "testingImages", filename),
                          fontsize, message, target_size, 'rgb(0,0,0)')
        testTextImages.append(np.array(images[0], dtype=float))
        i += 1
        if(i == 15):
            break

    # Convert datasets to NumPy arrays for use with TensorFlow
    textImages = np.array(textImages)
    expOutM1 = np.array(expOutM1)
    testTextImages = np.array(testTextImages)

    # Normalize the rgb values in the datasets to fall between 0 and 1
    textImages[:, :, :, :] /= 255
    testTextImages[:, :, :, :] /= 255

    if (loadModel == False):
        epochs = 300
        m = modelbuilder()
        # Save the model everytime validation loss hits new minimum
        mc = ModelCheckpoint(modelFilePath, monitor='val_loss', mode='min', verbose=1, save_best_only=True)
        # Train the model with the training dataset while reserving 20% of the data for validation
        history = m.fit(x=textImages[:size][:][:][:], y=expOutM1[:size][:][:][:], batch_size=100,
                        epochs=epochs, validation_split=0.2, use_multiprocessing=True, callbacks=mc)

        # Plot accuracy and loss results
        plt.figure(figsize=(10, 10))
        plt.subplot(1, 2, 1)
        epochs_range = range(epochs)
        plt.plot(epochs_range, history.history['accuracy'], label='Training Accuracy')
        plt.plot(epochs_range, history.history['val_accuracy'], label="validation Accuracy")
        plt.legend(loc='lower right')
        plt.title('Accuracy Results')
        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, history.history['loss'], label='Training Loss')
        plt.plot(epochs_range, history.history['val_loss'], label="Validation Loss")
        plt.legend(loc='upper right')
        plt.title('Loss Results')
        plt.savefig(os.path.join("inpainting", "results.png"))

    # Predict on the testing dataaset using the saved model
    saved_model = load_model(modelFilePath)
    results = saved_model.predict(testTextImages[:15][:][:][:])
    i = 0

    # Loop through the testing results and smooth the images based on the located text
    for images in results:
        gray = color.gray2rgb(images[:, :, 0])
        screen = np.array(gray)
        screen = screen / np.max(screen)
        for r in range(256):
            for c in range(256):
                if screen[r, c, 0] >= 0.92:
                    screen[r, c, 0] = 1
                    screen[r, c, 1] = 1
                    screen[r, c, 2] = 1
                else:
                    screen[r, c, 0] = 0
                    screen[r, c, 1] = 0
                    screen[r, c, 2] = 0
        # Save images with the text, detected text, and smoothed text
        io.imsave(os.path.join("inpainting", "bwThreshold", str(i + 1) + ".png"),
                  (screen * 255).astype('uint8'), check_contrast=False)
        io.imsave(os.path.join("inpainting", "testingImagesWithText", str(
            i + 1) + ".png"), (testTextImages[i] * 255).astype('uint8'))
        io.imsave(os.path.join("inpainting", "testingImagesDetectedText", str(
            i + 1) + ".png"), (gray * 255).astype('uint8'), check_contrast=False)
        smoothed = smooth(testTextImages[i], gray[:, :, 0] / np.max(gray))
        io.imsave(os.path.join("inpainting", "testingImagesRemovedText",
                               str(i + 1) + ".png"), (smoothed * 255).astype('uint8'))
        i += 1

    return

if __name__ == "__main__":
    '''
    To load our best model and predict with it, run 'inpainting(loadModel=True)'.
    - The training images are stored in inpainting/flower_photos
    - The testing images are stored in inpainting/testingImages
    - The testing images with overlaid text are stored in inpainting/testingImagesWithText
    - The testing detected text images are stored in inpainting/testingImagesDetectedText
    - The testing detected text images with 92% threshold are stored in inpainting/bwThreshold
    - The testing images with removed text are stored in inpainting/testingImagesWithRemovedText
    '''
    inpainting(loadModel=True)

    '''
    To train a new model and predict with it, run 'inpainting(modelFilePath='inpainting/m.h5', loadModel=False)'.
    - The newly trained model is stored as m.h5
    - The training images are stored in inpainting/flower_photos
    - The testing images are stored in inpainting/testingImages
    - The testing images with overlaid text are stored in inpainting/testingImagesWithText
    - The testing detected text images are stored in inpainting/testingImagesDetectedText
    - The testing detected text images with 92% threshold are stored in inpainting/bwThreshold
    - The testing images with removed text are stored in inpainting/testingImagesWithRemovedText
    '''
    # inpainting(modelFilePath='inpainting/m.h5', loadModel=False)
    