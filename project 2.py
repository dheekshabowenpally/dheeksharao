# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 13:52:06 2019

@author: Kumar
"""
from keras.models import Sequential
from keras.layers import Dense
import numpy

seed = 7
numpy.random.seed(seed)
dataset = numpy.loadtxt("C:/Users/Kumar/.spyder-py3/Deep Learning class/insurance.csv", delimiter=",")

X = dataset[:,0:4]
Y = dataset[:,4]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=seed)
model = Sequential()
model.add(Dense(12, input_dim=4, activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(6, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
model.fit(X_train, y_train, validation_data=(X_test,y_test), epochs=10, batch_size=10)
#acc: 51.12



#2 question to load data into json file
model_json = model.to_json()

with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("model.h2")
print("Saved model to disk")


json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h2")
print("Loaded model from disk")

# evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
#we need to  have features and labels for the model
dataset = numpy.loadtxt("C:/Users/Kumar/.spyder-py3/Deep Learning class/insurance.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:4]
Y = dataset[:,4]
score = loaded_model.evaluate(X, Y, verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
#acc: 49.48%



# 3rd question load weigght by yaml file
# serialize model to YAML
model_yaml = model.to_yaml()
with open("model.yaml", "w") as yaml_file:
    yaml_file.write(model_yaml)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")

# later...

#1st we need to load the model
# load YAML and create model
yaml_file = open('model.yaml', 'r')
loaded_model_yaml = yaml_file.read()
yaml_file.close()
loaded_model = model_from_yaml(loaded_model_yaml)
#2nd: load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

# evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
#next  step, we need to load the data
dataset = numpy.loadtxt("C:/Users/Kumar/.spyder-py3/Deep Learning class/insurance.csv", delimiter=",")

# split into input (X) and output (Y) variables
X = dataset[:,0:4]
Y = dataset[:,4]
score = loaded_model.evaluate(X, Y, verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
#acc: 49.48%





# 4th question Checkpoint the weights for best model on validation accuracy
# it overide the previous file in hdf5 format
#the previous method never overide the file since {epoch:02d} is unique in the file name
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint
import numpy
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# load pima indians dataset
dataset = numpy.loadtxt("C:/Users/Kumar/.spyder-py3/Deep Learning class/insurance.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:4]
Y = dataset[:,4]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=seed)

# create model
model = Sequential()
model.add(Dense(12, input_dim=4, kernel_initializer='uniform', activation='relu'))
model.add(Dense(10, kernel_initializer='uniform', activation='relu'))
model.add(Dense(8, kernel_initializer='uniform', activation='relu'))
model.add(Dense(6, kernel_initializer='uniform', activation='relu'))
model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# checkpoint
filepath="weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]
# Fit the model
model.fit(X_train, y_train, validation_data=(X_test,y_test), epochs=10, batch_size=10,callbacks=callbacks_list, verbose=0)

#acc:51.19
# 
# How to load and use weights from a checkpoint
# the program may crash or we do want this checkpoint as our starting point
#we must have two componets; first is the model structure
# we can specify it using keras or load from Json or YAML file
#2nd is the weights in hdf5 format;
from keras.models import Sequential
from keras.layers import Dense
import numpy
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=seed)

# create model using the codes or from JSON or YAML files
model = Sequential()
model.add(Dense(12, input_dim=4, kernel_initializer='uniform', activation='relu'))
model.add(Dense(10, kernel_initializer='uniform', activation='relu'))
model.add(Dense(8, kernel_initializer='uniform', activation='relu'))
model.add(Dense(6, kernel_initializer='uniform', activation='relu'))
model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))

# Compile model (required to make predictions)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# load weights from hdf5 file
model.load_weights("weights.best.hdf5")
print("Created model and loaded weights from file")
# load pima indians dataset
dataset = numpy.loadtxt("C:/Users/Kumar/.spyder-py3/Deep Learning class/insurance.csv", delimiter=",")

# split into input (X) and output (Y) variables
X = dataset[:,0:4]
Y = dataset[:,4]
# estimate accuracy on whole dataset using loaded weights
scores = model.evaluate(X, Y, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
#acc: 50.22%




