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
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Sequential, load_model

def modelbuilder():
    model = Sequential()
    model.add(Conv2D(1, (2, 2),strides = (1,1), activation='sigmoid', padding='same',input_shape=(256, 256, 3)))
    model.compile(optimizer='rmsprop', loss = 'mse', metrics=['accuracy'])
    
    model.summary()
    return model


def drawText(filepath, fontsize, text,target_size,color):
    image = Image.open(filepath)
    image = image.resize(target_size) #resize image 
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(font='arial.ttf', size=fontsize)
    size = font.getsize(text)
    x = randint(size[0]+16,255-16-size[0])
    y = randint(size[1]+16,255-16-size[1])
    print(color)
    draw.text((x,y), text, fill=color, font=font)
    #image.save('InpaintedImage.png')
    #image.close()
    
    #draw expected output from model 1
    
    x0 = x + 0.5*size[0] - 16
    y1 = y + 0.5*fontsize + 16
    x1 = x + 0.5*size[0] + 16 
    y0 = y + 0.5*fontsize - 16
    #draw input for model 2
    return image,(int(x0),int(y0),int(x1),int(y1))

def smooth(rgb_with_text, gray_detected_text):
    detectedPixels = []
    for r in range(256):
        for c in range(256):
            if gray_detected_text[r, c] > 0.90:
                detectedPixels.append([r, c])
                print([r, c])
    dP = np.array(detectedPixels)
    for loc in detectedPixels:
        r, c = loc[0], loc[1]
        average = [0, 0, 0]
        weight = 0
        if r + 1 < 256 and c + 1 < 256 and [r + 1, c + 1]:
            average[0] =+ rgb_with_text[r + 1, c + 1, 0]
            average[1] =+ rgb_with_text[r + 1, c + 1, 1]
            average[2] =+ rgb_with_text[r + 1, c + 1, 2]
            weight += 1
        if r + 1 < 256 and [r + 1, c]:
            average[0] =+ rgb_with_text[r + 1, c, 0]
            average[1] =+ rgb_with_text[r + 1, c, 1]
            average[2] =+ rgb_with_text[r + 1, c, 2]
            weight += 1
        if r + 1 < 256 and c - 1 >= 0 and [r + 1, c - 1]:
            average[0] =+ rgb_with_text[r + 1, c - 1, 0]
            average[1] =+ rgb_with_text[r + 1, c - 1, 1]
            average[2] =+ rgb_with_text[r + 1, c - 1, 2]
            weight += 1
        if c + 1 < 256 and [r, c + 1]:
            average[0] =+ rgb_with_text[r, c + 1, 0]
            average[1] =+ rgb_with_text[r, c + 1, 1]
            average[2] =+ rgb_with_text[r, c + 1, 2]
            weight += 1
        if c - 1 >= 0 and [r, c - 1]:
            average[0] =+ rgb_with_text[r, c - 1, 0]
            average[1] =+ rgb_with_text[r, c - 1, 1]
            average[2] =+ rgb_with_text[r, c - 1, 2]
            weight += 1
        if r - 1 >= 0 and c - 1 >= 0 and [r - 1, c - 1]:
            average[0] =+ rgb_with_text[r - 1, c - 1, 0]
            average[1] =+ rgb_with_text[r - 1, c - 1, 1]
            average[2] =+ rgb_with_text[r - 1, c - 1, 2]
            weight += 1
        if r - 1 >= 0 and [r - 1, c]:
            average[0] =+ rgb_with_text[r - 1, c, 0]
            average[1] =+ rgb_with_text[r - 1, c, 1]
            average[2] =+ rgb_with_text[r - 1, c, 2]
            weight += 1
        if r - 1 >= 0 and c + 1 < 256 and [r - 1, c + 1]:
            average[0] =+ rgb_with_text[r - 1, c + 1, 0]
            average[1] =+ rgb_with_text[r - 1, c + 1, 1]
            average[2] =+ rgb_with_text[r - 1, c + 1, 2]
            weight += 1
        if weight != 0:
            average[0] /= weight
            average[1] /= weight
            average[2] /= weight
            rgb_with_text[r, c] = average
    return rgb_with_text
#initialization variables
target_size = (256,256)

textImages = []
noTextImages = []
expOutM1 = []

testTextImages = []

size = 100
fontsize = 30

validation_message = "H"

i = 0
for filename in os.listdir('flower/flower_photos'):
    #message = ''.join(choices(ascii_uppercase, k = 1))  #randomize length of message and randomize message 
    validation_message = ''.join(choices(ascii_uppercase, k = 4))
    c = 'rgb(' + str(randint(0,255)) + ',' + str(randint(0,255)) + ',' + str(randint(0,255)) + ')'
    images = drawText('flower/flower_photos/' + filename,fontsize,validation_message,target_size,c)
    textImages.append(np.array(images[0],dtype = float))
    temp = np.zeros(shape = (256,256,3), dtype=float)
    x0 = images[1][0]
    y0 = images[1][1]
    x1 = images[1][2]
    y1 = images[1][3]    
    temp[y0:y1,x0:x1,:] = 1.0
    expOutM1.append(temp)
    noTextImages.append(np.array(Image.open('flower/flower_photos/' + filename).resize(target_size),dtype = float))
    i += 1
    if(i == size):
        break

b = 'rgb(0,0,0)'
i = 0
for filename in os.listdir('testingImages'):
    validation_message = ''.join(choices(ascii_uppercase, k = 4))
    c = 'rgb(' + str(randint(0,255)) + ',' + str(randint(0,255)) + ',' + str(randint(0,255)) + ')'
    images = drawText('testingImages/' + filename, fontsize, validation_message, target_size, b)
    images[0].save('testingImagesWithText/' + filename)
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

# plt.imshow(textImages[0])
# plt.show()
# plt.imshow(expOutM1[0])
# plt.show()

#m = modelbuilder()
#mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
#m.fit(x=textImages[:size][:][:][:],y=expOutM1[:size][:][:][:],batch_size=50,epochs=150,validation_split=0.2,use_multiprocessing=True,callbacks = mc)
saved_model = load_model('best_model.h5')
results = saved_model.predict(testTextImages[:15][:][:][:])

i = 0
for images in results: 
    gray = color.gray2rgb(images[:,:,0])
    plt.imshow(gray)
    plt.title("Max: " + str(np.max(gray)) + " Min: " + str(np.min(gray[:,:,0])))
    #plt.savefig('results/textDetect' + str(i) + ".png")
    plt.show()
    smoothed = smooth(testTextImages[i], gray[:,:,0] / np.max(gray))
    plt.imshow(smoothed)
    plt.show()
    i +=1
