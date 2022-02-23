# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 14:49:14 2021

@author: Djedouani
"""
from sklearn.metrics import classification_report,confusion_matrix
from tensorflow.keras.layers import Dense,Dropout,Conv2D,Activation,Flatten,MaxPooling2D
from  tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import  Adam
from tensorflow.keras.preprocessing.image import  ImageDataGenerator
import prepross
import cv2 as cv
import pickle
import numpy as np
from kerastuner.tuners import RandomSearch
from kerastuner.engine import hyperparameters

from sklearn.model_selection import train_test_split

db=pickle.load(open('basededonne.p','rb'))
labels=pickle.load(open('mylabels.p','rb'))
db=db/255
resizeddb=[]
# resizedval=[]
for i in range(len(db)):
    resizeddb.append( cv.resize(db[i],(50,50)))
# for i in range(len(validation)):
#    resizedval.append( cv.resize(validation[i].astype('uint16'),(64,64)))
resizeddb=np.array(resizeddb)
resizeddb=np.reshape(resizeddb,(174,50,50,1))
# resizedval=np.array(resizedval)
def buildmdl(hp):
    model = Sequential()
    model.add(Conv2D(64,kernel_size=(3,3),input_shape=(50,50,1)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2,2)))
    for i in range(hp.Int("n_layers",1,4)):
             model.add(Conv2D(hp.Int(f"conv_{i}_units",min_value=64,max_value=512,step=64),(3,3)))
             model.add(Activation("relu"))
    model.add(Flatten())
    model.add(Dropout(rate=0.1))
    model.add(Dense(10,activation="relu"))
    model.add(Dense(1, activation="sigmoid"))
       
  
# resizedval=np.reshape(resizedval,(58,64,64,1))
    model.compile(optimizer=Adam(lr=0.01),loss="binary_crossentropy",metrics=['accuracy'])
    return model



# datagen=ImageDataGenerator(rotation_range=30, width_shift_range=0.2,
# height_shift_range=0.2,
# horizontal_flip=True
#     )

#model.fit(resizeddb,labels,epochs=20,batch_size=32)
Xtrain,Xtest,Ytrain,Ytest=train_test_split(resizeddb,labels,test_size=0.2)
# mdl=buildmdl()
# mdl.fit(Xtrain, Ytrain, batch_size=32,epochs=1000,validation_data=(Xtest,Ytest))

tuner=RandomSearch(
    buildmdl,
    objective="val_accuracy",
    max_trials=1,
    executions_per_trial=1,
    overwrite=True,
       
    )

tuner.search(x=Xtrain,y=Ytrain,epochs=10,batch_size=32,validation_data=(Xtest,Ytest))

print(tuner.get_best_models()[0].summary())
k=tuner.results_summary()
print(tuner.results_summary())
print(tuner.get_best_hyperparameters()[0].values)
