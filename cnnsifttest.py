# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 12:04:00 2021

@author: Djedouani
"""

from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from tensorflow.keras.layers import Dense,Dropout,Conv2D,Activation,Flatten,MaxPooling2D,Conv1D,MaxPooling1D
from  tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import  Adam
from tensorflow.keras.preprocessing.image import  ImageDataGenerator
import prepross
import cv2 as cv
import pickle
import numpy as np
import sklearn.preprocessing as ps
from kerastuner.tuners import RandomSearch
from kerastuner.engine import hyperparameters
from sklearn.model_selection import train_test_split,StratifiedKFold
import extractionmethods as em
import classification as cl
from skfeature.function.similarity_based import fisher_score

db=pickle.load(open('basededonne.p','rb'))
labels=pickle.load(open('mylabels.p','rb'))

# for i in range(len(validation)):
#    resizedval.append( cv.resize(validation[i].astype('uint16'),(64,64)))
cropedlist_train=[]
for i in range(len(db)):
    cropedlist_train.append(prepross.crop(db[i],50,0))
    
des,train_features=em.extractfeaturesfromblocks(cropedlist_train,"sift")

    

minmax=ps.MinMaxScaler().fit(train_features)
X_train=minmax.transform(train_features)



# fvs=[]
# for i in range (len(db)):
#     hist=em.lbp(db[i],1, 8)
#     hist2=em.lbp(db[i],2, 16)
#     hist3=em.lbp(db[i],3, 24)
#     combined_hist=np.concatenate((hist,hist2,hist3))
#     fvs.append(combined_hist)


# minmax=ps.MinMaxScaler().fit(fvs)
# X_train=minmax.transform(fvs)
# X_train=np.reshape(X_train,(54,1,139))
meanaccuracies=[]
accuracy=[]
score=fisher_score.fisher_score(X_train, labels)
idx=fisher_score.feature_ranking(score)
for i in range (5,95,5):
    X_trainfs=X_train[:,idx[0:int(len(idx)*i/100)]]
    for train,test in StratifiedKFold(5).split(X_trainfs, labels):
    
        model = Sequential()
        model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(np.shape(X_trainfs)[1],1)))
        model.add(Dropout(0.1))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())
        model.add(Dense(32, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0001), metrics=['accuracy'])
        model.summary()
        Xtrainfs=np.reshape(X_trainfs[train,:],(np.shape(X_trainfs[train,:])[0],np.shape(X_trainfs[train,:])[1],1))
        Xtestfs=np.reshape(X_trainfs[test,:],(np.shape(X_trainfs[test,:])[0],np.shape(X_trainfs[train,:])[1],1))
        model.fit(Xtrainfs, labels[train], batch_size=len(Xtrainfs),epochs=100)
        y_predictions=np.round(model.predict(Xtestfs))
        accuracy.append(accuracy_score(y_predictions, labels[test]))
    meanaccuracies.append(np.average(accuracy))
