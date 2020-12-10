import numpy as np
import os
import PIL
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from skimage import color
import matplotlib.pyplot as plt
import pathlib

def modelbuilder(numlayers):
    model = Sequential()
    #model.add(layers.experimental.preprocessing.Resizing(input_shape = (180,180,1)))
    # model.add(layers.Conv2D(64,3,activation='relu',padding='same', use_bias = True, kernel_regularizer = tf.keras.regularizers.L1L2(l1 = 0.01, l2= 0.01)))
    # i = 0 
    # while i < numlayers: #
    #     model.add(layers.Conv2D(64,3,activation='relu', padding='same', use_bias = True, kernel_regularizer = tf.keras.regularizers.L1L2(l1 = 0.01, l2= 0.01)))
    #     i += 1
    # model.add(layers.Conv2D(64,3,activation='relu', padding='same', use_bias = True, kernel_regularizer = tf.keras.regularizers.L1L2(l1 = 0.01, l2= 0.01)))
    # model.add(layers.Conv2D(3,3,activation='relu', padding='same', use_bias = True, kernel_regularizer = tf.keras.regularizers.L1L2(l1 = 0.01, l2= 0.01)))
    model.add(layers.InputLayer(input_shape=(100, 100, 1)))
    model.add(keras.layers.Conv2D(128, kernel_size=3, activation='relu', padding='same'))
    model.add(keras.layers.Conv2D(128, kernel_size=3, activation='relu', padding='same'))
    model.add(keras.layers.Conv2D(128, kernel_size=3, activation='relu', padding='same'))
    model.add(keras.layers.Conv2D(128, kernel_size=3, activation='relu', padding='same'))
    model.add(keras.layers.Conv2D(3, kernel_size=3, activation='relu', padding='same'))

    model.compile(optimizer='sgd', loss = 'mse', metrics=['accuracy'])
    model.summary()
    return model

trainGray = []
for filename in os.listdir('trainingImages/'):
    trainGray.append(keras.preprocessing.image.img_to_array(keras.preprocessing.image.load_img(path = 'trainingImages/'+filename, color_mode ='grayscale', target_size=(100,100))))
trainGray = np.array(trainGray, dtype=float)
print(trainGray.shape)
trainGray /= 255.0

trainLab = []
for filename in os.listdir('trainingImages/'):
    trainLab.append(color.rgb2lab(keras.preprocessing.image.img_to_array(keras.preprocessing.image.load_img(path = 'trainingImages/'+filename, color_mode ='rgb', target_size=(100,100)))))
trainLab = np.array(trainLab, dtype=float)
print(trainLab.shape)
print(np.max(trainLab))
print(np.min(trainLab))
trainLab /= 255.0

plt.figure(figsize=(10,10))
for i in range(6):
    plt.subplot(4,4,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(trainGray[i,:,:,0])
    plt.xlabel("RGB Grayscale")
for i in range(6):
    plt.subplot(4,4,i+7)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(trainLab[i,:,:,0])
    plt.xlabel("Lab Grayscale")
plt.show()


trainColor = []
for filename in os.listdir('trainingImages/'):
    trainColor.append(keras.preprocessing.image.img_to_array(keras.preprocessing.image.load_img(path = 'trainingImages/'+filename, color_mode ='rgb', target_size=(100,100))))
trainColor = np.array(trainColor, dtype=float)
print(trainColor.shape)
trainColor /= 255.0

m = modelbuilder(1)
for i in range(4):
    output = m.fit(trainGray[:1], trainColor[:1], epochs = 50, batch_size=1, use_multiprocessing=True, verbose=2)
    out = m.predict(trainGray[:1])
    plt.figure()
    plt.imshow(out[0])
    plt.colorbar()
    plt.grid(False)
    plt.show()

print(out.shape)
plt.figure()
plt.imshow(out[0])
plt.colorbar()
plt.grid(False)
plt.show()

''' 
Structure: 
1) load in images from google storage API
2) convert images to bw
3) rescale 
4) loop with different models
    4a) different layers? we definitely need 2 conv2D - 1st hidden layer and output layer
    4b) 
    4c) plot on graph (training error and validation error)
5) choose best model.
6) profit!
'''