# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 13:46:32 2019

@author: Kumar
"""
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
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
modelresult=model.fit(X_train, y_train, validation_data=(X_test,y_test), epochs=100, batch_size=10)
scores = model.evaluate(X, Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

print(modelresult.history.keys())

plt.plot(modelresult.history['acc'])
plt.plot(modelresult.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(modelresult.history['loss'])
plt.plot(modelresult.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()



from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
from keras.layers import Dropout

import numpy
from keras.optimizers import SGD
from sklearn.model_selection import StratifiedKFold
from keras.wrappers.scikit_learn import KerasClassifier
from keras.constraints import maxnorm
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold



seed = 7
numpy.random.seed(seed)
dataset = numpy.loadtxt("C:/Users/Kumar/.spyder-py3/Deep Learning class/insurance.csv", delimiter=",")

X = dataset[:,0:4]
Y = dataset[:,4]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=seed)

def create_model():
    model = Sequential()
    model.add(Dropout(0.25, input_shape=(4,)))
    model.add(Dense(12, input_dim=4,kernel_initializer='normal', activation='relu',kernel_constraint=maxnorm(3)))
    model.add(Dense(10,kernel_initializer='normal',activation='relu',kernel_constraint=maxnorm(3)))
    model.add(Dense(8,kernel_initializer='normal', activation='relu',kernel_constraint=maxnorm(3)))
    model.add(Dense(6,kernel_initializer='normal', activation='relu',kernel_constraint=maxnorm(3)))
    model.add(Dense(1, kernel_initializer='normal',activation='sigmoid'))
    sgd = SGD(lr=0.1, momentum=0.9, decay=0.0, nesterov=False)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


estimator = KerasClassifier(build_fn=create_model, epochs=200, batch_size=5, verbose=0)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)


results = cross_val_score(estimator, X, Y, cv=kfold)
print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
#Accuracy: 51.80% (2.82%)




from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
from keras.layers import Dropout

import numpy
from keras.optimizers import SGD
from sklearn.model_selection import StratifiedKFold
from keras.wrappers.scikit_learn import KerasClassifier
from keras.constraints import maxnorm
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold



seed = 7
numpy.random.seed(seed)
dataset = numpy.loadtxt("C:/Users/Kumar/.spyder-py3/Deep Learning class/insurance.csv", delimiter=",")

X = dataset[:,0:4]
Y = dataset[:,4]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=seed)

def create_model():
    model = Sequential()
    model.add(Dense(12, input_dim=4,kernel_initializer='normal', activation='relu',kernel_constraint=maxnorm(3)))
    model.add(Dropout(0.25))
    model.add(Dense(10,kernel_initializer='normal',activation='relu',kernel_constraint=maxnorm(3)))
    model.add(Dropout(0.25))
    model.add(Dense(8,kernel_initializer='normal', activation='relu',kernel_constraint=maxnorm(3)))
    model.add(Dropout(0.25))
    model.add(Dense(6,kernel_initializer='normal', activation='relu',kernel_constraint=maxnorm(3)))
    model.add(Dense(1, kernel_initializer='normal',activation='sigmoid'))
    model.add(Dropout(0.25))
    sgd = SGD(lr=0.1, momentum=0.9, decay=0.0, nesterov=False)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


estimator = KerasClassifier(build_fn=create_model, epochs=200, batch_size=5, verbose=0)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)


results = cross_val_score(estimator, X, Y, cv=kfold)
print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
#Accuracy: 50.75% (0.65%)


#There is no much difeerence in accuracy by dropping input layer or hidden layer 
#instead we can change the dropout % to find change in accuracy








