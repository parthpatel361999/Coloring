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


def augment(image):
    image = tf.cast(image,tf.float32)
    image = tf.image.rgb_to_grayscale(image)
    return image
def modelbuilder(numlayers):
    model = Sequential()
    model.add(layers.experimental.preprocessing.Rescaling(1./255, input_shape = (180,180,1)))
    model.add(layers.Conv2D(64,3,activation='tanh',padding='same', use_bias = True, kernel_regularizer = tf.keras.regularizers.L1L2(l1 = 0.01, l2= 0.01)))
    i = 0 
    while i < numlayers: #
        model.add(layers.Conv2D(64,3,activation='tanh', padding='same', use_bias = True, kernel_regularizer = tf.keras.regularizers.L1L2(l1 = 0.01, l2= 0.01)))
        i += 1
    
    model.add(layers.Conv2D(3,3,activation='tanh', padding='same', use_bias = True, kernel_regularizer = tf.keras.regularizers.L1L2(l1 = 0.01, l2= 0.01)))
    
    model.compile(optimizer='adam',loss = 'mse')
    #model.summary()
    return model

#load in images in
# TODO currently working on this
path = os.getcwd() + "\\flower"

batch_size = 32
img_height = img_width = 180

# train_ds = tf.keras.preprocessing.image_dataset_from_directory(
#     pathlib.Path(path), label_mode = None, validation_split=0.2,
#     seed=123, image_size = (img_height,img_width),
#     subset = "training"
# )

# train_ds_2 = (
#     train_ds
#     .map(augment) #add autotuning
# )


# val_ds = tf.keras.preprocessing.image_dataset_from_directory(
#     pathlib.Path(path),label_mode = None, validation_split= 0.2,
#     seed=123, image_size = (img_height,img_width),
#     batch_size = batch_size,
#     subset = "validation"
# )


# #check if properly grayscaled
# plt.figure(figsize=(10,10))

# for images in train_ds.take(1):
#     for i in range(9):
#         ax = plt.subplot(3,3, i + 1)
#         plt.imshow(images[i].numpy().astype("uint8"))
#         plt.title("Is she grayscale?")
#         plt.axis('off')
# plt.show()

# for images in train_ds_2.take(1):
#     for i in range(9):
#         ax = plt.subplot(3,3, i + 1)
#         plt.imshow(images[i].numpy().astype("uint8").reshape((180,180)),cmap="gray")
#         plt.title("Is she grayscale?")
#         plt.axis('off')
# plt.show()

X = []
for filename in os.listdir('trainingImages/'):
    X.append(img_to_array(load_img(path = 'trainingImages/'+filename, color_mode ='grayscale', target_size=(180,180))))
X = np.array(X, dtype=float)
print(X.shape)

Y = []
for filename in os.listdir('trainingImages/'):
    Y.append(img_to_array(load_img(path = 'trainingImages/'+filename, color_mode ='rgb', target_size=(180,180))))
Y = np.array(Y, dtype=float)
print(Y.shape)

m = modelbuilder(1)
m.summary()
output = m.fit(X, Y, epochs = 5)

# Set up train and test data
split = int(0.95*len(X))
Xtrain = X[:split]
Xtrain = 1.0/255*Xtrain



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