from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os
import PIL
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from skimage import color, io, transform
from keras.layers import Conv2D, UpSampling2D, Dropout,Dense,MaxPooling2D
import matplotlib.pyplot as plt
import pathlib
from random import randint,choices
from string import ascii_uppercase

def modelbuilder():
    model = Sequential()
    
    model.add(Conv2D(64, (16, 16),strides = (1,1), activation='relu', padding='same',input_shape=(256, 256, 3)))
    model.add(Dense(128,activation="relu"))
    model.add(Dense(64,activation="relu"))
    model.add(Dense(64,activation="relu"))
    model.add(Dense(32,activation="relu"))
    model.add(Conv2D(1, (32, 32), strides = (1,1) ,activation="softmax", padding='same'))
    model.compile(optimizer='adam', loss = 'binary_crossentropy', metrics=['accuracy'])
    
    model.summary()
    return model


def drawText(filepath, fontsize, text,target_size):
    image = Image.open(filepath)
    image = image.resize(target_size) #resize image 
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(font='arial.ttf', size=fontsize)
    size = font.getsize(text)
    x = randint(size[0]+16,255-16-size[0])
    y = randint(size[1]+16,255-16-size[1])
    color = 'rgb(0,0,0)' #black

    draw.text((x,y), text, fill=color, font=font)
    #image.save('InpaintedImage.png')
    #image.close()
    
    #draw expected output from model 1
    
    x0 = x + 0.5*size[0] - 8
    y1 = y + 0.5*fontsize + 8
    x1 = x + 0.5*size[0] + 8
    y0 = y + 0.5*fontsize - 8
    print((int(x0),int(y0),int(x1),int(y1)))
    #draw input for model 2
    image3 = Image.open(filepath).resize(target_size)

    draw = ImageDraw.Draw(image3)
    draw.rectangle(xy = [(x0,y0),(x1,y1)], fill = "#FFFFFF",outline = "white")
    return image,(int(x0),int(y0),int(x1),int(y1)),image3


#initialization variables
target_size = (256,256)

textImages = []
noTextImages = []
textImagesRectangle = []
expOutM1 = []

testTextImages = []

size = 1
fontsize = 15

validation_message = "v"

i = 0
for filename in os.listdir('flower/flower_photos'):
    #message = ''.join(choices(ascii_uppercase, k = 1))  #randomize length of message and randomize message 
    message = "H"
    images = drawText('flower/flower_photos/' + filename,fontsize,message,target_size)
    textImages.append(np.array(images[0],dtype = float))
    temp = np.zeros(shape = (256,256,3), dtype=float)
    x0 = images[1][0]
    y0 = images[1][1]
    x1 = images[1][2]
    y1 = images[1][3]    
    temp[y0:y1,x0:x1,:] = 1.0
    expOutM1.append(temp)
    textImagesRectangle.append(np.array(images[2],dtype = float))
    noTextImages.append(np.array(Image.open('flower/flower_photos/' + filename).resize(target_size),dtype = float))
    i += 1
    if(i == size):
        break

i = 0
for filename in os.listdir('testingImages'):
    images = drawText('testingImages/' + filename,fontsize,validation_message,target_size)
    images[0].save('testingImagesWithText/' + filename)
    testTextImages.append(np.array(images[0],dtype = float))
    i += 1
    if(i == 1):
        break

textImages = np.array(textImages)
textImagesRectangle = np.array(textImagesRectangle)
noTextImages = np.array(noTextImages)
expOutM1 = np.array(expOutM1)

testTextImages = np.array(testTextImages)

textImages[:,:,:,:] /= 255
noTextImages[:,:,:,:] /= 255
textImagesRectangle[:,:,:,:] /= 255


plt.imshow(textImages[0])
plt.show()
plt.imshow(expOutM1[0])
plt.show()
plt.imshow(textImagesRectangle[0])
plt.show()

testTextImages[:,:,:,:] /= 255
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

m.fit(x=textImages[:size][:][:][:],y=expOutM1[:size][:][:][:],batch_size=1,epochs=10,validation_split=0,use_multiprocessing=True)
results = m.predict(textImages[:1][:][:][:])

'''
i = 0
for images in results: 
    plt.imshow(images)
    plt.savefig('results/' + str(i) + ".png")
    plt.show()
    i +=1
'''