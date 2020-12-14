from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os
import PIL
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from skimage import color, io
import matplotlib.pyplot as plt
import pathlib
from random import randint, choices
from string import ascii_uppercase
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Sequential, load_model

def modelbuilder():
    model = Sequential()
    model.add(layers.Conv2D(1, (2, 2), strides = (1,1), activation='sigmoid', padding='same', input_shape=(256, 256, 3)))
    model.compile(optimizer='rmsprop', loss = 'mse', metrics=['accuracy'])
    model.summary()
    return model

def drawText(filepath, fontsize, text,target_size,color):
    image = Image.open(filepath)
    image = image.resize(target_size)
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(font='arial.ttf', size=fontsize)
    size = font.getsize(text)
    x = randint(size[0]+16,255-16-size[0])
    y = randint(size[1]+16,255-16-size[1])
    draw.text((x,y), text, fill=color, font=font)
    x0 = x + 0.5*size[0] - 16
    y1 = y + 0.5*fontsize + 16
    x1 = x + 0.5*size[0] + 16 
    y0 = y + 0.5*fontsize - 16
    return image,(int(x0),int(y0),int(x1),int(y1))

def getNeighbors(r, c, buffer):
    neighbors = [(r - buffer, c - buffer), (r - buffer, c), (r - buffer, c + buffer), (r, c - buffer),
                 (r, c + buffer), (r + buffer, c - buffer), (r + buffer, c), (r + buffer, c + buffer)]
    section = []
    for neighbor in neighbors:
        nR, nC = neighbor
        if nR >= 0 and nR < 256 and nC >= 0 and nC < 256:
            section.append([nR, nC])
    return section

def smooth(rgb_with_text, gray_detected_text):
    detectedPixels = []
    for r in range(256):
        for c in range(256):
            if gray_detected_text[r, c] > 0.90:
                detectedPixels.append([r, c])
    length = len(detectedPixels)
    for i in range(length):
        neighbors = getNeighbors(detectedPixels[i][0], detectedPixels[i][1], 1)
        for neighbor in neighbors:
            if neighbor not in detectedPixels:
                detectedPixels.append(neighbor)
    for loc in detectedPixels:
        neighbors = getNeighbors(loc[0], loc[1], 3)
        averagedPixel = [0, 0, 0]
        numPixels = 0
        for neighbor in neighbors:
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

#initialization variables
target_size = (256,256)
textImages = []
noTextImages = []
expOutM1 = []
testTextImages = []
size = 100
fontsize = 30

i = 0
for filename in os.listdir('flower/flower_photos'): 
    message = ''.join(choices(ascii_uppercase, k = 4))
    c = 'rgb(' + str(randint(0, 255)) + ',' + str(randint(0, 255)) + ',' + str(randint(0, 255)) + ')'
    images = drawText('flower/flower_photos/' + filename, fontsize, message, target_size, c)
    textImages.append(np.array(images[0], dtype = float))
    temp = np.zeros(shape = (256, 256, 3), dtype = float)
    x0 = images[1][0]
    y0 = images[1][1]
    x1 = images[1][2]
    y1 = images[1][3]    
    temp[y0:y1,x0:x1,:] = 1.0
    expOutM1.append(temp)
    noTextImages.append(np.array(Image.open('flower/flower_photos/' + filename).resize(target_size), dtype = float))
    i += 1
    if(i == size):
        break

b = 'rgb(0,0,0)'
i = 0
for filename in os.listdir('testingImages'):
    message = ''.join(choices(ascii_uppercase, k = 4))
    c = 'rgb(' + str(randint(0, 255)) + ',' + str(randint(0, 255)) + ',' + str(randint(0, 255)) + ')'
    images = drawText('testingImages/' + filename, fontsize, message, target_size, b)
    testTextImages.append(np.array(images[0], dtype = float))
    i += 1
    if(i == 15):
        break

textImages = np.array(textImages)
noTextImages = np.array(noTextImages)
expOutM1 = np.array(expOutM1)
testTextImages = np.array(testTextImages)

textImages[:,:,:,:] /= 255
noTextImages[:,:,:,:] /= 255
testTextImages[:,:,:,:] /= 255

# epochs = 200
# m = modelbuilder()
# #es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
# mc = ModelCheckpoint('test_model_1.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)
# history = m.fit(x=textImages[:size][:][:][:], y=expOutM1[:size][:][:][:], batch_size=25, epochs=epochs, validation_split=0.2, use_multiprocessing=True, callbacks=mc)

#plot history

# plt.figure(figsize = (10,10))
# plt.subplot(1,2,1)
# epochs_range = range(epochs)
# plt.plot(epochs_range, history.history['accuracy'], label = 'Training Accuracy')
# plt.plot(epochs_range, history.history['val_accuracy'], label = "validation Accuracy")
# plt.legend(loc = 'lower right')
# plt.title('Accuracy Results')

# plt.subplot(1,2,2)
# plt.plot(epochs_range, history.history['loss'], label = 'Training Loss')
# plt.plot(epochs_range, history.history['val_loss'], label = "Validation Loss")
# plt.legend(loc = 'upper right')
# plt.title('Loss Results')

# plt.savefig("results.png")
# plt.show()

#run predictions
saved_model = load_model('proto1_model.h5')
results = saved_model.predict(testTextImages[:15][:][:][:])
i = 0
for images in results: 
    gray = color.gray2rgb(images[:, :, 0])

    io.imsave('testingImagesWithText/' + str(i + 1) + ".png", (testTextImages[i] * 255).astype('uint8'))
    io.imsave('testingImagesDetectedText/' + str(i + 1) + ".png", (gray * 255).astype('uint8'), check_contrast=False)
    smoothed = smooth(testTextImages[i], gray[:, :, 0] / np.max(gray))
    io.imsave('testingImagesRemovedText/' + str(i + 1) + ".png", (smoothed * 255).astype('uint8'))

    # plt.imshow(gray)
    # plt.savefig('results/textDetected' + str(i + 1) + ".png")
    # plt.show()
    # smoothed = smooth(testTextImages[i], gray[:, :, 0] / np.max(gray))
    # plt.imshow(smoothed)
    # plt.savefig('results/textRemoved' + str(i + 1) + ".png")
    # plt.show()
    i +=1
