# -*- coding: utf-8 -*-
"""
Created on Sat Apr 10 14:26:48 2021

@author: Djedouani
"""
import prepross 
import extractionmethods as em
import  cv2 as cv
import numpy as np
import pickle
import matplotlib.pyplot as plt
import classification as cl
from sklearn.utils import shuffle
import sklearn.preprocessing as ps
from sklearn.model_selection import train_test_split
from  sklearn.feature_selection import SelectKBest,f_classif,RFECV,chi2,SelectPercentile
from skfeature.function.similarity_based import fisher_score
from sklearn import svm
from sklearn.metrics import roc_curve
import time
# validationsets=['normed_val_8bit_clahe.p','normed_val_8bit_minmax.p']
# db=pickle.load(open('normed_db_8bit_clahe.p','rb'))
# blind=pickle.load(open('normed_val_8bit_clahe.p','rb'))
# labels=pickle.load(open('labels.p','rb'))
# val_labels=pickle.load(open('val_labels.p','rb'))
# db=np.array(db).astype('uint8')
# blind=np.array(blind).astype('uint8')
db=pickle.load(open('basededonne.p','rb'))
labels=pickle.load(open('mylabels.p','rb'))
# basededonne,labels=shuffle(basededonne,labels)
# class0=prepross.readimgsfromfolder('dataaugmentation/class 0/*')
# class1=prepross.readimgsfromfolder('dataaugmentation/class 1/*')
# blind=prepross.readimgsfromfolder('blind/*')
# blindlabels=np.array([0,1,0,0,0,0,0,0,0,0,1,0,1,1,0,0,
#                       1,0,0,1,0,1,1,1,1,0,1,0,1,1,1,0,
#                       0,0,1,1,1,1,0,1,1,0,1,0,0,1,0,1,
#                       1,1,1,1,1,0,0,1,0,0])
# db,labels=prepross.getnormeddb(class0,class1)
# db,labels=shuffle(db,labels)
cropedlist_train=[]
cropedlist_test=[]
for i in range(len(db)):
    cropedlist_train.append(prepross.crop(db[i],200,0))
# for i in range(len(blind)):
#     cropedlist_test.append(prepross.crop(blind[i],100,0))
# train_features1=em.extractfeaturesfromblocks(cropedlist_train,"glcm")
# train_features2=em.extractfeaturesfromblocks(cropedlist_train,"lbp")
# train_features=np.concatenate((train_features1,train_features2),axis=1)
des,train_features=em.extractfeaturesfromblocks(cropedlist_train,"orb")
# test_features=em.extractfeaturesfromblocks(cropedlist_test,'lbp')
    
# scaler=ps.StandardScaler()
# transformer=scaler.fit(train_features)
# X_train=transformer.transform(train_features)
# X_test=transformer.transform(test_features)
minmax=ps.MinMaxScaler().fit(train_features)
X_train=minmax.transform(train_features)
# scaler2=ps.MinMaxScaler().fit(test_features)
# X_test=minmax.transform(test_features)
# cl.testmodel(grid, lbp_tst, val_labels)
   
grid,bestscore=cl.hyperparamstuning(1, 5, X_train, labels)


accuracieslist=[]
fsscore=0;
bestscore=0
for i in range(100):
    x_train,x_test,y_train,y_test=train_test_split(X_train,labels,test_size=0.2,stratify=labels)
    model=svm.SVC()
    model.fit(x_train,y_train)
    score=model.score(x_test,y_test)
    if bestscore<score:
        bestscore=score
        bestmodel=model
# cl.testmodel(bestmodel,x_test,y_test)
# cl.testmodel(bestmodel,X_test,blindlabels)
accuracieslist=[]
fsscore=0;
bestscore=0
for k in range(5,91,5):
    score=fisher_score.fisher_score(X_train, labels)
    idx=fisher_score.feature_ranking(score)
    X_trainfs=X_train[:,idx[0:int(k/100*X_train.shape[1])]]
    grid,bestscore=cl.hyperparamstuning(1, 5, X_trainfs, labels)
    if fsscore<bestscore:
        fsscore=bestscore;
        best_params=grid.best_params_
        bestmodel=grid
    accuracieslist.append(bestscore)   
print(best_params)  
print(fsscore)

# start=time.time()
# bestacc,percentage=0,0
# for i in range (5,91,5):
#     fs=SelectPercentile(score_func=chi2,percentile=i).fit(X_train,labels)
#     X_trainfs=fs.transform(X_train)
#     grid,bestmeanacc=cl.hyperparamstuning(0, 5, X_trainfs, labels)
#     if bestacc<bestmeanacc:
#         bestacc=bestmeanacc
#         percentage=i
# end=time.time()
# print(end - start)
# print("bestacc " + str(bestacc) +" %")
# print(str(percentage) + " %")

# print(max(accuracieslist))
# fs=SelectPercentile(score_func=f_classif,percentile=10).fit(X_train2,labels)
# X_train=fs.transform(X_train2)
# model=create_ANN_Model((20,50),'nadam')
# X_test=fs.transform(x_test2)
# y_prediction =model.predict(x_test2)
# y_prediction=np.round(y_prediction)
# print(confusion_matrix(val_labels,y_prediction))
# print(classification_report(val_labels, y_prediction))
# pickle.dump(basededonne,open('basededonne.p','wb'))
# pickle.dump(labels,open('mylabels.p','wb'))
