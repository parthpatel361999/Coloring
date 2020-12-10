import numpy as np
import os
import PIL
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from skimage import color, io, transform
from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D
import matplotlib.pyplot as plt
import pathlib

target_size = (256,256)

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
    
    
    model.add(layers.InputLayer(input_shape=(256, 256, 1)))
    # model.add(keras.layers.Conv2D(128, kernel_size=3, activation='relu', padding='same'))
    # model.add(keras.layers.Conv2D(128, kernel_size=3, activation='relu', padding='same'))
    # model.add(keras.layers.Conv2D(128, kernel_size=3, activation='relu', padding='same'))
    # model.add(keras.layers.Conv2D(128, kernel_size=3, activation='relu', padding='same'))
    # model.add(keras.layers.Conv2D(2, kernel_size=3, activation='relu', padding='same'))

    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', strides=2))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same', strides=2))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same', strides=2))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(2, (3, 3), activation='tanh', padding='same'))
    model.add(UpSampling2D((2, 2)))

    model.compile(optimizer='rmsprop', loss = 'mse', metrics=['accuracy'])
    model.summary()
    return model

# trainGray = []
# for filename in os.listdir('trainingImages/'):
#     trainGray.append(keras.preprocessing.image.img_to_array(keras.preprocessing.image.load_img(path = 'trainingImages/'+filename, color_mode ='grayscale', target_size=target_size)))
# trainGray = np.array(trainGray, dtype=float)
# print(trainGray.shape)
# trainGray /= 255.0
i = 0
trainLab = []
for filename in os.listdir('flower/flower_photos'):
    trainLab.append(color.rgb2lab(transform.resize(1.0/255*io.imread('flower/flower_photos/' + filename), target_size)))
    i += 1
    if(i == 24):
        break
trainLab = np.array(trainLab, dtype=float)
trainLab[:,:,:,0] /= 100
trainLab[:,:,:,1] /= 128
trainLab[:,:,:,2] /= 128
print("trainlab size")
print(np.max(trainLab[0,:,:,0]))
print(np.min(trainLab[0,:,:,0]))
print(np.max(trainLab[0,:,:,1]))
print(np.min(trainLab[0,:,:,1]))
print(np.max(trainLab[0,:,:,2]))
print(np.min(trainLab[0,:,:,2]))

# trainColor = []
# for filename in os.listdir('trainingImages/'):
#     trainColor.append(keras.preprocessing.image.img_to_array(keras.preprocessing.image.load_img(path = 'trainingImages/'+filename, color_mode ='rgb', target_size=(100,100))))
# trainColor = np.array(trainColor, dtype=float)
# print(trainColor.shape)
# trainColor /= 255.0

m = modelbuilder(1)
output = m.fit(trainLab[:30,:,:,:1], trainLab[:30,:,:,1:], epochs = 300, batch_size=5, use_multiprocessing=True, validation_split=0.2, verbose=2)
out = m.predict(trainLab[:30,:,:,:1])

for i in range(len(out)):
    img = np.zeros((256,256,3))
    img[:,:,0] = trainLab[i][:,:,0] * 100
    img[:,:,1:] = out[i] * 128
    img = color.lab2rgb(img)
    
    plt.figure()
    plt.imshow(img) 
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