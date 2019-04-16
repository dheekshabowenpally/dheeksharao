# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 14:34:20 2019

@author: Kumar
"""

import time 
from PIL import Image #Python Images Library;pillow
import matplotlib.pyplot as plt
import numpy as np #4d arrays of color images
import warnings
#train and validation set/test set
from sklearn.model_selection import train_test_split
import keras
#Flatten convert 2d to 1d; Dropout to address overfit
from keras.layers import Dense, Flatten, Dropout #Flatten convert 2- dimensional matrix to 1 vector
#CNN 
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
#The following packages help us automatically load all pictures
#into a 4d array
from os import listdir #list  all files and directiory in a given path
from os.path import isfile, join #test file or directory

#start clock
t0 = time.time()
# input image dimensions for CNN
#we need to convert the origin picture to this size based on pillow package
img_x, img_y = 128, 128 #we determine the size first, then we convert the give image to this size


from numpy.random import seed
seed(1)

from tensorflow import set_random_seed
set_random_seed(2)

#mypath = (r"C:\Users\genre\OneDrive\Desktop\MS_DataSceince\DeepLearning\DataSets\Fruits\")
mypath = ("C:/Users/Kumar/.spyder-py3/Deep Learning class/fruits/fruits/")

onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

applelabel = 0 
orangelabel = 1
imagelst = []

y = np.zeros(len(onlyfiles), dtype = np.int8)

idx = 0
for item in onlyfiles:
    file = mypath + item
    print("file={0}".format(file))
    img = Image.open(file)
    print("The original image size = {0} times {1}={2}".format(img.size[0], img.size[1],img.size[0]* img.size[1] ))
    img=img.resize((img_x, img_y), Image.ANTIALIAS)
    print("The new image size = {0} times {1}={2}".format(img.size[0], img.size[1],img.size[0]* img.size[1] ))
    rgbm = np.array(img)
    print(rgbm.shape)
    #begin handle 4th channel for transparency; discard it and convert to 3 channels with RGB
    if rgbm.shape[2] == 4:
        rgbm=rgbm[:,:,:3]
        print("Warning!: discard 4th channel of {0}".format(file))
    print(rgbm.shape)
    imagelst.append(rgbm)
    
    if "apple" in item:
        y[idx] = applelabel
    elif "org" in item:
        y[idx] = orangelabel
    else:
        warnings.warn("The object is not in given class!!")
    idx = idx + 1

X = np.stack(imagelst, axis=0)
#print out the size of X
X.shape
#step 4  train test set split

#80/20 split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=2018)

print('x_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')    

fig, axs = plt.subplots(2,5, figsize=(15, 6), facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace = .5, wspace=.001)
#map the label 0: tree; 1: grass
pic_label=["apple", "orange"]
#Return a contiguous flattened array. 2d -> 1d; easier for us to write code
#using 1 index instead of 2
axs = axs.ravel()
#by the mapping above, we only need to use one loop since it is 1d
#instead of a nested loops (two loops) for original 2d structure
for i in range(10):
    axs[i].imshow(X_train[i,:,:,:])
    axs[i].set_title(pic_label[y_train[i]])

#step 6 : CNN

batch_size = 10
num_classes = 2 # 2 classes; tree or grass
epochs = 10 #one epochs means passing the full dataset/training set to the neorual network.

# the data is a 4D tensor - (sample_number, x_img_size, y_img_size, num_channels)
#the features are already 4 dimension
#input images have the same size after they were resized using pillow package
input_shape = (img_x, img_y, 3)

print("The data types of array elements of X_train and X_test are {0} and {1} respectively".format(X_train.dtype,
      X_test.dtype))
# convert the data to the right type: float for decimals
#python is a "strong" type language, we need to  convert int to float mannually

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
#normalize it and divide it by 255, which is the largest  number 8-bit
#2^8-1=255
#get a number between 0 and 1
X_train /= 255 #X_train = X_train/255
X_test /= 255 #X_test = X_test/255
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
#
model.add(Conv2D(64, kernel_size=(5, 5), strides=(1, 1),
                 activation='relu',
                 input_shape=input_shape))

model.add(Dropout(0.2))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Conv2D(128, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
#
model.add(Flatten())

model.add(Dense(50, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(num_classes, activation='softmax'))

#it is classification not regression
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer= keras.optimizers.Adam(),
              metrics=['accuracy'])


model.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(X_test, y_test)
          )
score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

t1 = time.time()
total_n = t1-t0
print("time = {0}".format(total_n))

#val_acc: 0.7500
#Test loss: 0.9073666930198669
#Test accuracy: 0.75
#time = 50.51527190208435


model = Sequential()
#
model.add(Conv2D(64, kernel_size=(5, 5), strides=(1, 1),
                 activation='relu',
                 input_shape=input_shape))

model.add(Dropout(0.2))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Conv2D(128, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))
#
model.add(Flatten())

model.add(Dense(50, activation='relu'))
model.add(Dropout(0.3))

model.add(Dense(num_classes, activation='softmax'))

#it is classification not regression
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer= keras.optimizers.Adam(),
              metrics=['accuracy'])


model.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(X_test, y_test)
          )
score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


#val_acc: 0.6250
#Test loss: 0.6174356937408447
#Test accuracy: 0.625

model = Sequential()
#
model.add(Conv2D(64, kernel_size=(5, 5), strides=(1, 1),
                 activation='relu',
                 input_shape=input_shape))

model.add(Dropout(0.2))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Conv2D(128, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))

model.add(Conv2D(128, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))
#
model.add(Flatten())

model.add(Dense(50, activation='relu'))
model.add(Dropout(0.3))

model.add(Dense(num_classes, activation='softmax'))

#it is classification not regression
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer= keras.optimizers.Adam(),
              metrics=['accuracy'])


model.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(X_test, y_test)
          )
score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

#val_acc: 0.5000
#Test loss: 0.982393205165863
#Test accuracy: 0.5
#Test loss: 0.982393205165863
#Test accuracy: 0.5


#0.2 dropout has more accuracy than 0.3 

        
        
























