import numpy as np
import os
import PIL
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt
import pathlib
#build model first
# TODO Have not tested yet
def modelbuilder(numlayers):
    model = Sequential()
    model.add(layers.Conv2D(64,3,activation='tanh',use_bias = True, kernel_regularizer = tf.keras.regularizers.L1L2(l1 = 0.01, l2= 0.01)))
    i = 0 
    while i < numlayers: #
        model.add(layers.Conv2D(64,3,activation='tanh',use_bias = True, kernel_regularizer = tf.keras.regularizers.L1L2(l1 = 0.01, l2= 0.01)))
        i += 1
    
    model.add(layers.Conv2D(3,3,activation='tanh',use_bias = True, kernel_regularizer = tf.keras.regularizers.L1L2(l1 = 0.01, l2= 0.01)))
    model.compile(optimizer='adam',loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'])

    return model

#load in images in
# TODO currently working on this
path = os.getcwd() + "\\flower"

batch_size = 32
img_height = img_width = 180

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    pathlib.Path(path), label_mode = None, color_mode='grayscale',
    seed=123, image_size = (img_height,img_width),
    batch_size = batch_size,
    validation_split = 0.2,
    subset = "training"
)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    pathlib.Path(path),color_mode='grayscale', label_mode = None,
    seed=123, image_size = (img_height,img_width),
    batch_size = batch_size,
    validation_split = 0.2,
    subset = "validation"
)

#check if properly grayscaled
plt.figure()
for images in train_ds.take(1):
    for i in range(3):
        plt.imshow(images[i].numpy().astype("float").reshape(180,180))
        plt.title("Is she grayscale?")
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