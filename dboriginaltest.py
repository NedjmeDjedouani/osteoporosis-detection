# -*- coding: utf-8 -*-
"""
Created on Wed May 26 16:03:31 2021

@author: Djedouani
"""

import prepross 
import extractionmethods as em
import classification as cl
import matplotlib.pyplot as plt
import numpy as np
import pickle 
import  matplotlib.pyplot as plt
import sklearn.preprocessing as ps
from  sklearn.feature_selection import SelectKBest,RFECV,chi2,SelectPercentile
from skfeature.function.similarity_based import fisher_score

import time
import lightgbm 
basededonne=pickle.load(open('basededonne.p','rb'))
labels=pickle.load(open('mylabels.p','rb'))
img1=prepross.readimgsfromfolder('class1/*')
img=(img1[1]/256).astype('uint8')
plt.hist(img.flatten(),bins=256,range=(0,255))
""" lbp """

fvs=[]
for i in range (len(basededonne)):
    hist=em.lbp(basededonne[i],1, 8)
    hist2=em.lbp(basededonne[i],2, 16)
    hist3=em.lbp(basededonne[i],3, 24)
    combined_hist=np.concatenate((hist,hist2,hist3))
    fvs.append(combined_hist)


minmax=ps.MinMaxScaler().fit(fvs)
X_train=minmax.transform(fvs)

gird,gridscore=cl.hyperparamstuning(1,5,X_train,labels )

""" glcm """


fvs=[]
for i in range (len(basededonne)):
    fv=em.glcm_features(basededonne[i],[1,3,5],[0,np.pi/4,np.pi/2])
    fvs.append(fv)
    
minmax=ps.MinMaxScaler().fit(fvs)
X_train=minmax.transform(fvs)
cl.hyperparamstuning(1,5,X_train,labels )


""" sift """

deslist=[]
for i in range (len(basededonne)):
    des=em.extractsiftdes(basededonne[i],100)
    deslist.append(des)
fvs=em.kmeans_clustering(deslist,300)    
minmax=ps.MinMaxScaler().fit(fvs)
X_train=minmax.transform(fvs)
cl.hyperparamstuning(1,5,X_train,labels )

""" orb """

deslist=[]
for i in range (len(basededonne)):
    des=em.extractOrbdes(basededonne[i],100)
    deslist.append(des)
    
fvs=em.kmeans_clustering(deslist,300)    
minmax=ps.MinMaxScaler().fit(fvs)
X_train=minmax.transform(fvs)
cl.hyperparamstuning(1,5,X_train,labels)
# """ chi feature selection """
# start=time.time()
# bestacc,percentage=0,0
# for i in range (5,91,5):
#     fs=SelectPercentile(score_func=chi2,percentile=i).fit(X_train,labels)
#     X_trainfs=fs.transform(X_train)
#     grid,bestmeanacc=cl.hyperparamstuning(1, 5, X_trainfs, labels)
#     if bestacc<bestmeanacc:
#         bestacc=bestmeanacc
#         percentage=i
# end=time.time()
# print(end - start)
# print("bestacc " + str(bestacc) +" %")
# print(str(percentage) + " %")

""" fisher feature selection """
fsscore,percentage,accuracieslist=0,0,[]
start=time.time()

for k in range(5,91,5):
    score=fisher_score.fisher_score(X_train, labels)
    idx=fisher_score.feature_ranking(score)
    X_trainfs=X_train[:,idx[0:int(k/100*X_train.shape[1])]]
    grid,bestscore=cl.hyperparamstuning(1, 5, X_trainfs, labels)
    if fsscore<bestscore:
        fsscore=bestscore;
        best_params=grid.best_params_
        bestmodel=grid
        percentage=k
    accuracieslist.append(bestscore) 
end=time.time()
print(end - start)
    
print(fsscore)
print(percentage)    
print(best_params)  
# cl.lgbmlightmodel(5, X_train, labels)

