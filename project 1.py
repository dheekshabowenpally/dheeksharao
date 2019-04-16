# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 15:23:00 2019

@author: Kumar
"""
#Automatic
from keras.models import Sequential
from keras.layers import Dense
import numpy

numpy.random.seed(7)
dataset = numpy.loadtxt("C:/Users/Kumar/.spyder-py3/Deep Learning class/insurance.csv", delimiter=",")

X = dataset[:,0:4]
Y = dataset[:,4]

model = Sequential()
model.add(Dense(12, input_dim=4, activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X, Y, validation_split=0.2, epochs=150, batch_size=10)


#accuracy automatic:  0.5103 *100= 51.03%

#manual
from sklearn.model_selection import train_test_split
seed = 7
numpy.random.seed(seed)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=seed)
model = Sequential()
model.add(Dense(12, input_dim=4, activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
model.fit(X_train, y_train, validation_data=(X_test,y_test), epochs=150, batch_size=10)
#accuracy manual: 0.5042*100=50.42%


# K-fold Cross Validation (CV);
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
import numpy


def create_model():
    model = Sequential()
    model.add(Dense(12, input_dim=4, activation='relu'))
    model.add(Dense(10,activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
seed = 7
numpy.random.seed(seed)
dataset = numpy.loadtxt("C:/Users/Kumar/.spyder-py3/Deep Learning class/insurance.csv", delimiter=",")
X = dataset[:,0:4]
Y = dataset[:,4]

model=KerasClassifier(build_fn=create_model, epochs=150, batch_size=10, verbose=0)
kfold=StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)


results = cross_val_score(model, X, Y, cv=kfold)
print(results.mean())
print(results.std())

#accuracy or mean for k-fold 0.5007519708255186*100=50.07519708255186%

#Ans: Accuracy for Automatic validation is more when compared to manual and Kfold validation
#Automatic validation is the best model







