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


#initialization variables
target_size = (256,256)

textImages = []
noTextImages = []
expOutM1 = []

testTextImages = []

size = 300
fontsize = 30

validation_message = "H"

i = 0
for filename in os.listdir('flower/flower_photos'):
    #message = ''.join(choices(ascii_uppercase, k = 1))  #randomize length of message and randomize message 
    message = "H"
    c = 'rgb(' + str(randint(0,255)) + ',' + str(randint(0,255)) + ',' + str(randint(0,255)) + ')' #black
    images = drawText('flower/flower_photos/' + filename,fontsize,message,target_size,c)
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

i = 0
for filename in os.listdir('testingImages'):

    images = drawText('testingImages/' + filename,fontsize,validation_message,target_size,'rgb(0,0,0)')
    images[0].save('testingImagesWithText/' + filename)
    testTextImages.append(np.array(images[0],dtype = float))
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

plt.imshow(textImages[0])
plt.show()
plt.imshow(expOutM1[0])
plt.show()




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

epochs = 50
m = modelbuilder()
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
mc = ModelCheckpoint('best_model.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)
history = m.fit(x=textImages[:size][:][:][:],y=expOutM1[:size][:][:][:],batch_size=25,epochs=epochs,validation_split=0.2,use_multiprocessing=True,callbacks=[es, mc])

#plot history
plt.figure(figsize = (10,10))
plt.subplot(1,2,1)
epochs_range = range(epochs)
plt.plot(epochs_range, history.history['accuracy'], label = 'Training Accuracy')
plt.plot(epochs_range, history.history['val_accuracy'], label = "validation Accuracy")
plt.legend(loc = 'lower right')
plt.title('Accuracy Results')

plt.subplot(1,2,2)
plt.plot(epochs_range, history.history['loss'], label = 'Training Loss')
plt.plot(epochs_range, history.history['val_loss'], label = "Validation Loss")
plt.legend(loc = 'upper right')
plt.title('Loss Results')

plt.savefig("results.png")
plt.show()

#run predictions
saved_model = load_model('best_model.h5')
results = saved_model.predict(testTextImages[:15][:][:][:])
i = 0
for images in results: 
    print(images)
    plt.imshow(color.gray2rgb(images[:,:,0]))
    plt.savefig('results/' + str(i) + ".png")
    plt.show()
    i +=1
