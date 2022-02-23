# -*- coding: utf-8 -*-
"""
Created on Sat May 29 10:09:57 2021

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
from sklearn.metrics import plot_roc_curve,plot_confusion_matrix,accuracy_score,classification_report
import time


class0=prepross.readimgsfromfolder('dataaugmentation/class 0/*')
class1=prepross.readimgsfromfolder('dataaugmentation/class 1/*')
db,labels=prepross.getnormeddb(class0, class1)
dbclahe=prepross.clahefun(db,0.2,(40,40))
dbclahe=prepross.to8bit(dbclahe)
cropedlist_train=[]
cropedlist_test=[]
for i in range(len(db)):
    cropedlist_train.append(prepross.crop(dbclahe[i],50,0))


fvs=[]
for i in range (len(dbclahe)):
    hist=em.lbp(dbclahe[i],1, 8)
    hist2=em.lbp(dbclahe[i],2, 16)
    hist3=em.lbp(dbclahe[i],3, 24)
    combined_hist=np.concatenate((hist,hist2,hist3))
    fvs.append(combined_hist)

des,train_features=em.extractfeaturesfromblocks(cropedlist_train,"sift")

deslist=[]
for i in range (len(dbclahe)):
    des=em.extractOrbdes(dbclahe[i],100)
    deslist.append(des)
    
fvs=em.kmeans_clustering(deslist,100)    
minmax=ps.MinMaxScaler().fit(fvs)
X_train=minmax.transform(fvs)
    
minmax=ps.MinMaxScaler().fit(train_features)
X_train=minmax.transform(train_features)

trainx,testx,trainy,testy = train_test_split(X_train,labels,test_size=0.2,stratify=labels,random_state=20)
mdl=svm.SVC(C=1,gamma=0.1,kernel='rbf').fit(trainx,trainy)
plot_confusion_matrix(mdl,testx,testy)
predictions=mdl.predict(testx)
accuracy_score(testy,predictions)
plot_roc_curve(mdl,testx,testy,)
plt.xlabel('y')
print(classification_report(testy,predictions))

# grid,bestscore=cl.hyperparamstuning(0, 5, X_train, labels)
pickle.dump(mdl,open('bestmdl.p','wb'))
