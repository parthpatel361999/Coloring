import os
import random
from time import time

import keras
import numpy as np
import tensorflow as tf
from keras.callbacks import TensorBoard
from keras.engine import Layer
from keras.layers import (Activation, Conv2D, Conv2DTranspose, Conv3D, Dense,
                          Dropout, Flatten, Input, InputLayer, Reshape,
                          UpSampling2D, concatenate, merge)
from keras.layers.normalization import BatchNormalization
from keras.models import Model, Sequential
from keras.preprocessing import image
from keras.preprocessing.image import (ImageDataGenerator, array_to_img,
                                       img_to_array, load_img)
from PIL import Image, ImageFile
from skimage.color import gray2rgb, lab2rgb, rgb2gray, rgb2lab
from skimage.io import imsave
from skimage.transform import resize

from common import checkQuality2, convertToGrayscale, getImagePixels

# target_size = (256, 256)

path = "flower"
train_datagen = ImageDataGenerator(rescale=1. / 255)
train = train_datagen.flow_from_directory(path, target_size=(256, 256), batch_size=400, class_mode=None)

X = []
Y = []
for img in train[0]:
    try:
        lab = rgb2lab(img)
        X.append(lab[:, :, 0])
        Y.append(lab[:, :, 1:] / 128)
    except:
        print('error')
X = np.array(X)
Y = np.array(Y)
X = X.reshape(X.shape+(1,))
print(X.shape)
print(Y.shape)

# Encoder
encoder_input = Input(shape=(256, 256, 1,))
encoder_output = Conv2D(64, (3, 3), activation='relu', padding='same', strides=2)(encoder_input)
encoder_output = Conv2D(128, (3, 3), activation='relu', padding='same')(encoder_output)
encoder_output = Conv2D(128, (3, 3), activation='relu', padding='same', strides=2)(encoder_output)
encoder_output = Conv2D(256, (3, 3), activation='relu', padding='same')(encoder_output)
encoder_output = Conv2D(256, (3, 3), activation='relu', padding='same', strides=2)(encoder_output)
encoder_output = Conv2D(512, (3, 3), activation='relu', padding='same')(encoder_output)
encoder_output = Conv2D(512, (3, 3), activation='relu', padding='same')(encoder_output)
encoder_output = Conv2D(256, (3, 3), activation='relu', padding='same')(encoder_output)
# Decoder
decoder_output = Conv2D(128, (3, 3), activation='relu', padding='same')(encoder_output)
decoder_output = UpSampling2D((2, 2))(decoder_output)
decoder_output = Conv2D(64, (3, 3), activation='relu', padding='same')(decoder_output)
decoder_output = UpSampling2D((2, 2))(decoder_output)
decoder_output = Conv2D(32, (3, 3), activation='relu', padding='same')(decoder_output)
decoder_output = Conv2D(16, (3, 3), activation='relu', padding='same')(decoder_output)
decoder_output = Conv2D(2, (3, 3), activation='tanh', padding='same')(decoder_output)
decoder_output = UpSampling2D((2, 2))(decoder_output)
model = Model(inputs=encoder_input, outputs=decoder_output)


model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
model.fit(X, Y, validation_split=0.2, epochs=100, batch_size=40)

test_path = 'testingImages/'
test = os.listdir(test_path)
color_me = []
for i, imgName in enumerate(test):
    img = img_to_array(load_img(test_path + imgName))
    img = resize(img, (256, 256))
    color_me.append(img)
color_me = np.array(color_me, dtype=float)
color_me = rgb2lab(1.0/255*color_me)[:, :, :, 0]
color_me = color_me.reshape(color_me.shape+(1,))
print(color_me.shape)

output = model.predict(color_me)
output = output * 128

# Output colorizations
for i in range(len(output)):
    result = np.zeros((256, 256, 3))
    result[:, :, 0] = color_me[i][:, :, 0]
    result[:, :, 1:] = output[i]
    # result = np.array(result, dtype=np.uint8)
    imsave("doublyImprovedResults/result"+str(i)+".png", lab2rgb(result))

# for i in range(len(output)):
#     oP = getImagePixels("testingImages", str(i) + ".png")
#     nP = getImagePixels("doublyImprovedResults", "result" + str(i) + ".png")
#     print(checkQuality2(oP, nP))

# def modelbuilder():
#     model = Sequential()
#     model.add(layers.experimental.preprocessing.RandomFlip("horizontal", input_shape=(256, 256, 1)))
#     model.add(layers.experimental.preprocessing.RandomRotation(0.1))
#     model.add(layers.experimental.preprocessing.RandomZoom(0.1))
#     model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
#     model.add(Dropout(0.1))
#     model.add(Conv2D(64, (3, 3), activation='relu', padding='same', strides=2))
#     model.add(Dropout(0.1))
#     model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
#     model.add(Conv2D(128, (3, 3), activation='relu', padding='same', strides=2))
#     model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
#     model.add(Conv2D(256, (3, 3), activation='relu', padding='same', strides=2))
#     model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
#     model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
#     model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
#     model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
#     model.add(UpSampling2D((2, 2)))
#     model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
#     model.add(UpSampling2D((2, 2)))
#     model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
#     model.add(Conv2D(2, (3, 3), activation='tanh', padding='same'))
#     model.add(UpSampling2D((2, 2)))

#     model.compile(optimizer='rmsprop', loss='mse', metrics=['accuracy'])
#     model.summary()
#     return model


# i = 0
# trainLab = []
# for filename in os.listdir('flower/flower_photos'):
#     trainLab.append(color.rgb2lab(transform.resize(1.0/255*io.imread('flower/flower_photos/' + filename), target_size)))
#     i += 1
#     if(i == 400):
#         break
# trainLab = np.array(trainLab, dtype=float)
# trainLab[:, :, :, 0] /= 100
# trainLab[:, :, :, 1] /= 128
# trainLab[:, :, :, 2] /= 128
# print("trainlab size")
# print(np.max(trainLab[0, :, :, 0]))
# print(np.min(trainLab[0, :, :, 0]))
# print(np.max(trainLab[0, :, :, 1]))
# print(np.min(trainLab[0, :, :, 1]))
# print(np.max(trainLab[0, :, :, 2]))
# print(np.min(trainLab[0, :, :, 2]))

# # trainColor = []
# # for filename in os.listdir('trainingImages/'):
# #     trainColor.append(keras.preprocessing.image.img_to_array(keras.preprocessing.image.load_img(path = 'trainingImages/'+filename, color_mode ='rgb', target_size=(100,100))))
# # trainColor = np.array(trainColor, dtype=float)
# # print(trainColor.shape)
# # trainColor /= 255.0

# testLab = []
# for filename in os.listdir('testingImages/'):
#     testLab.append(color.rgb2lab(transform.resize(1.0/255*io.imread('testingImages/' + filename), target_size)))
# testLab = np.array(testLab, dtype=float)
# testLab[:, :, :, 0] /= 100
# testLab[:, :, :, 1] /= 128
# testLab[:, :, :, 2] /= 128


# m = modelbuilder()
# m.save("model")
# epochs = 600
# batch_size = 40
# output = m.fit(trainLab[:400, :, :, :1], trainLab[:400, :, :, 1:], epochs=epochs,
#                batch_size=batch_size, use_multiprocessing=True, validation_split=0.2)


# # Performance Graph
# acc = output.history['accuracy']
# val_acc = output.history['val_accuracy']

# loss = output.history['loss']
# val_loss = output.history['val_loss']

# iterations = range(epochs)

# plt.figure(figsize=(10, 10))
# plt.subplot(1, 2, 1)
# plt.plot(iterations, acc, label='Training Accuracy')
# plt.plot(iterations, val_acc, label='Validation Accuracy')
# plt.legend(loc='lower right')
# plt.title('Accuracy')

# plt.subplot(1, 2, 2)
# plt.plot(iterations, loss, label='Training Loss')
# plt.plot(iterations, val_loss, label='Validation Loss')
# plt.legend(loc='upper right')
# plt.title('Loss')
# built = "Results from " + str(epochs) + " Epochs at " + str(batch_size) + " of " + str(i) + " Images"
# plt.savefig(built + '.png')
# # plt.show()


# out = m.predict(testLab[:30, :, :, :1])

# for i in range(len(out)):
#     img = np.zeros((256, 256, 3))
#     img[:, :, 0] = testLab[i][:, :, 0] * 100
#     img[:, :, 1:] = out[i] * 128
#     img = color.lab2rgb(img)

#     plt.figure()
#     plt.imshow(img)
#     plt.colorbar()
#     plt.grid(False)
#     plt.savefig(str(i) + '.png')
#     plt.show()
