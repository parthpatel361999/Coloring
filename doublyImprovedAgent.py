import numpy as np
import os
import PIL
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt
import pathlib
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.layers import Conv2D


def augment(image):
    image = tf.cast(image,tf.float32)
    image = tf.image.rgb_to_grayscale(image)
    return image
def modelbuilder(numlayers):
    model = Sequential()
    model.add(layers.experimental.preprocessing.Resizing(180, 180, input_shape = (180,180,1)))
    # model.add(layers.Conv2D(64,3,activation='relu',padding='same', use_bias = True, kernel_regularizer = tf.keras.regularizers.L1L2(l1 = 0.01, l2= 0.01)))
    # i = 0 
    # while i < numlayers: #
    #     model.add(layers.Conv2D(64,3,activation='relu', padding='same', use_bias = True, kernel_regularizer = tf.keras.regularizers.L1L2(l1 = 0.01, l2= 0.01)))
    #     i += 1
    # model.add(layers.Conv2D(64,3,activation='relu', padding='same', use_bias = True, kernel_regularizer = tf.keras.regularizers.L1L2(l1 = 0.01, l2= 0.01)))
    # model.add(layers.Conv2D(3,3,activation='relu', padding='same', use_bias = True, kernel_regularizer = tf.keras.regularizers.L1L2(l1 = 0.01, l2= 0.01)))
    
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(3, (3, 3), activation='tanh', padding='same'))

    model.compile(optimizer='adam', loss = 'mse')
    #model.summary()
    return model

#load in images in
# TODO currently working on this
path = os.getcwd() + "\\flower"

batch_size = 32
img_height = img_width = 180

X = []
for filename in os.listdir('trainingImages/'):
    X.append(img_to_array(load_img(path = 'trainingImages/'+filename, color_mode ='grayscale', target_size=(180,180))))
X = np.array(X, dtype=float)
print(X.shape)
X /= 255.0

Y = []
for filename in os.listdir('trainingImages/'):
    Y.append(img_to_array(load_img(path = 'trainingImages/'+filename, color_mode ='rgb', target_size=(180,180))))
Y = np.array(Y, dtype=float)
print(Y.shape)
Y /= 255.0

m = modelbuilder(1)
m.summary()
output = m.fit(X, Y, validation_split = 0.2, epochs = 3, batch_size=10)

out = m.predict(X[0:2])

print(out.shape)
print(out[0,0,0])
plt.figure()
plt.imshow(out[0])
plt.colorbar()
plt.grid(False)
plt.show()

print(out.shape)
print(out[0,0,0])
plt.figure()
plt.imshow(out[1])
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