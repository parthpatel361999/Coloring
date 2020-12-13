from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os
import PIL
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from skimage import color, io, transform
from keras.layers import Conv2D, UpSampling2D, Dropout, Dense
import matplotlib.pyplot as plt
import pathlib
from random import randint,choices
from string import ascii_letters, digits

def modelbuilder():
    model = Sequential()
    model.add(layers.experimental.preprocessing.RandomRotation(0.05, input_shape = (256,256,3)))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(Dense(64))#, activation='tanh'))
    #model.add(Conv2D(3, (3, 3), activation='relu', padding='same'))
    model.add(Dense(3))#, activation='sigmoid'))
 

    model.compile(optimizer='adam', loss = 'mse', metrics=['accuracy'])
    model.summary()
    return model

def drawText(filepath, fontsize, text,target_size):
    image = Image.open(filepath)
    image = image.resize(target_size) #resize image 
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(font='arial.ttf', size=fontsize)
    size = font.getsize(text)
    (x,y) = (randint(fontsize,255-size[0]),randint(0,255-size[1]))
    color = 'rgb(0,0,0)' #black

    draw.text((x,y), text, fill=color, font=font)
    #image.save('InpaintedImage.png')
    #image.close()
    
    #draw expected output from model 1
    image2 = Image.new(mode = "RGB",size = target_size, color = (255,255,255))
    draw = ImageDraw.Draw(image2)
    draw.text((x,y), text, fill=color, font=font)


    #draw input for model 2
    image3 = Image.open(filepath).resize(target_size)
    draw = ImageDraw.Draw(image3)
    draw.rectangle(xy = [(x,y),(x+size[0],y+size[1])], fill = "#FFFFFF",outline = "white")
    
    return image,image2,image3


#initialization variables
target_size = (256,256)

textImages = []
noTextImages = []
textImagesRectangle = []
expOutM1 = []

testTextImages = []

size = 600
fontsize = 15

validation_message = "val"

i = 0
for filename in os.listdir('flower/flower_photos'):
    message = "message" #randomize length of message and randomize message 
    images = drawText('flower/flower_photos/' + filename, fontsize, message, target_size)
    textImages.append(np.array(images[0],dtype = float))
    expOutM1.append(np.array(images[1],dtype = float))
    textImagesRectangle.append(np.array(images[2],dtype = float))
    noTextImages.append(np.array(Image.open('flower/flower_photos/' + filename).resize(target_size),dtype = float))
    i += 1
    if(i == size):
        break

i = 0
for filename in os.listdir('testingImages'):
    images = drawText('testingImages/' + filename,fontsize,validation_message,target_size)
    testTextImages.append(np.array(images[0],dtype = float))
    io.imsave("testingWithText/" + str(i) + ".png", testTextImages[i])
    i += 1
    if(i == 15):
        break

textImages = np.array(textImages)
textImagesRectangle = np.array(textImagesRectangle)
noTextImages = np.array(noTextImages)
expOutM1 = np.array(expOutM1)

testTextImages = np.array(testTextImages)

textImages[:,:,:,:] /= 255
noTextImages[:,:,:,:] /= 255
expOutM1[:,:,:,:] /= 255
textImagesRectangle[:,:,:,:] /= 255
testTextImages[:,:,:,:] /= 255

# plt.imshow(textImages[0])
# plt.show()
# plt.imshow(noTextImages[0])
# plt.show()
# plt.imshow(testTextImages[0])
# plt.show()

#need to find smallest rectange that contains text, then fill that box with white

'''
Flow:
input: RGB image with text
output: white image with black text in correct location (ie, is there text on this image? where?)

output will have shape (size,256,256,3). need to find the spot of text on the image:
    1) convert back to Image
    2) scan through pixel coordinates (x,y) to find nonwhite pixel, record corner
    3) 

m2.input: (RGB image with text) - output = RGB image with white box covering text
m2.output: rectangle of pixles that corresponds to the white box

Final output:
m2.input + m2.output

Training: 
model 1: have all info
model 2: RGB image with white box around text. 
'''



m = modelbuilder()
m.fit(x=textImages[:50][:][:][:],y=noTextImages[:50][:][:][:],batch_size=10,epochs=40,use_multiprocessing=True, validation_split=0.1)
results = m.predict(testTextImages[:30][:][:][:])


i = 0
for images in results:
    io.imsave('results/' + str(i) + ".png", images)
    i +=1
