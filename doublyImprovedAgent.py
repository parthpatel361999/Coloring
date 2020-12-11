import numpy as np
import os
import PIL
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from skimage import color, io, transform
from keras.layers import Conv2D, UpSampling2D, Dropout
import matplotlib.pyplot as plt
import pathlib
from random import randint
target_size = (256,256)



def modelbuilder(numlayers):
    model = Sequential()    
    model.add(layers.experimental.preprocessing.RandomFlip("horizontal",input_shape=(256, 256, 1)))
    model.add(layers.experimental.preprocessing.RandomRotation(0.1))
    model.add(layers.experimental.preprocessing.RandomZoom(0.1))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(Dropout(0.1))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', strides=2))
    model.add(Dropout(0.1))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same', strides=2))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same', strides=2))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
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

i = 0
trainLab = []
for filename in os.listdir('flower/flower_photos'):
    trainLab.append(color.rgb2lab(transform.resize(1.0/255*io.imread('flower/flower_photos/' + filename), target_size)))
    i += 1
    if(i == 1):
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

testLab = []
for filename in os.listdir('testingImages/'):
    testLab.append(color.rgb2lab(transform.resize(1.0/255*io.imread('testingImages/' + filename), target_size)))
testLab = np.array(testLab, dtype=float)
testLab[:,:,:,0] /= 100
testLab[:,:,:,1] /= 128
testLab[:,:,:,2] /= 128


m = modelbuilder(1)
'''
output = m.fit(trainLab[:30,:,:,:1], trainLab[:30,:,:,1:], epochs = 300, batch_size=5, use_multiprocessing=True, validation_split=0.2, verbose=2)
out = m.predict(testLab[:30,:,:,:1])

for i in range(len(out)):
    img = np.zeros((256,256,3))
    img[:,:,0] = testLab[i][:,:,0] * 100
    img[:,:,1:] = out[i] * 128
    img = color.lab2rgb(img)
    
    plt.figure()
    plt.imshow(img) 
    plt.colorbar()
    plt.grid(False)
    plt.show()
'''