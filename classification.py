# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 14:40:59 2021

@author: Djedouani
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold,GridSearchCV,train_test_split,StratifiedKFold
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn import svm
import numpy as np
import tensorflow as tf 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Activation
from tensorflow.keras.optimizers import  Adam
from sklearn.neighbors import KNeighborsClassifier
# import lightgbm as lgb

def hyperparamstuning(modeltype,k,fvs,labels):
    kfold=StratifiedKFold(k)
    if modeltype ==0 :
        print('SVM')
        param_grid={'C':[0.1,1,10,100],'gamma':[0.001,0.01,0.1,1],'kernel':['rbf','sigmoid']}
        grid=GridSearchCV(svm.SVC(),param_grid,refit=True,cv=kfold.split(fvs,labels))
    elif modeltype==1 :
        print('RandomForest')

        param_grid={'n_estimators':[10,50,100,500],'criterion':['gini','entropy']}
        grid=GridSearchCV(RandomForestClassifier(),param_grid,refit=True,cv=kfold.split(fvs,labels))
    elif modeltype==2 :
        print('KNN')

        param_grid={'leaf_size':list(range(1,10)),'n_neighbors':list(range(1,15,2)),'p':[1,2]}
        grid=GridSearchCV(KNeighborsClassifier(),param_grid,refit=True,cv=kfold.split(fvs,labels))
    grid.fit(fvs,labels)
    print(grid.best_score_)
    return grid,grid.best_score_ 

def testmodel(model,ytest,ylabels):
        predictions=model.predict(ytest)
        print(classification_report(ylabels,predictions))  
        print(confusion_matrix(ylabels,predictions))

def Kfoldcrossvalidation(modeltype,featurevectors,labels,k):
 
    if modeltype==0:
        clf=svm.SVC(cv=k)
    elif modeltype==1:
        clf=RandomForestClassifier(cv=k)
    elif modeltype==2:
        clf=KNeighborsClassifier(cv=k)
   # scores=[]

    clf.fit(featurevectors,labels,)
# def lgbmlightmodel(k,fvs,labels):
#     kfold=StratifiedKFold(k)
#     acclist=[]
#     for train,test in kfold.split(fvs,labels):
#        train_data=lgb.Dataset(fvs[train],label=labels[train])
#        param = {'num_leaves':200, 'objective':'binary','max_depth':7,'learning_rate':.05,'max_bin':200}
#        param['metric'] = ['auc', 'binary_logloss']
#        mdl=lgb.train(param,train_data,200)
#        predictions=np.round(mdl.predict(fvs[test]))
#        print(predictions)
#        acc=accuracy_score(labels[test],predictions)
#        print(acc)
#        acclist.append(acc)
 
#     return np.average(acclist)


