# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 14:37:51 2021

@author: Djedouani
"""
import numpy as np
import prepross 
import extractionmethods as em
import  classification as cl
import sklearn.preprocessing as ps

from  sklearn.feature_selection import SelectKBest,f_classif,RFECV,chi2,SelectPercentile
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.models import Sequential
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.layers import Dense,Activation,Dropout
from sklearn.model_selection import KFold,GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
from skfeature.function.similarity_based import fisher_score

from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
import pickle


db=pickle.load(open('basededonne.p','rb'))
labels=pickle.load(open('mylabels.p','rb'))

cropedlist_train=[]
for i in range(len(db)):
    cropedlist_train.append(prepross.crop(db[i],50,0))



method='lbp'
Xtrainf1=em.extractfeaturesfromblocks(cropedlist_train,method)


method='glcm'
Xtrainf2=em.extractfeaturesfromblocks(cropedlist_train,method)


Xtrain_combined=np.hstack((Xtrainf1,Xtrainf2))




def svmdlwfs(X_train,labels,features_factor):      #save model with feature selection 
    nf=int(X_train.shape[1]*features_factor)
    
    bestacc=0
    for k in range(20,nf,10):
        fs=SelectKBest(score_func=f_classif,k=k)
        fstransformer=fs.fit(X_train,labels)
        X_train_new=fstransformer.transform(X_train)
        scaler=ps.StandardScaler().fit(X_train_new)
        X_train_new=scaler.transform(X_train_new)
        model,acc=cl.hyperparamstuning(0, 10, X_train_new, labels)
        if acc>bestacc:
            bestacc=acc
            transformer=fstransformer
            scalertransformer=scaler
            bestmdl=model
    pickle.dump(transformer,open("transformer.p","wb")) 
    pickle.dump(bestmdl.best_estimator_,open("bestmodel.p","wb"))
    pickle.dump(scalertransformer,open("scalertransformer.p",'wb'))
    
    
svmdlwfs(Xtrainf1,labels,0.7)    
    
    
# pickle.dump(Xvalf1,open("X_test.p",'wb'))       
    
  

# scaler=ps.StandardScaler().fit(Xtrain_combined)
# X_train_new=scaler.transform(Xtrain_combined)
minmax=ps.MinMaxScaler().fit(Xtrain_combined)
X_train2=minmax.transform(Xtrain_combined)
# m=RFECV(RandomForestClassifier(),scoring="accuracy")
# m.fit(X_train_new,labels)
# m.score(X_test_new,val_labels)

accuracieslist=[]
for k in range(1,70,1):
    fs=SelectPercentile(score_func=f_classif,percentile=k).fit(X_train2,labels)
    X_train=fs.transform(X_train2)
    model,acc=cl.hyperparamstuning(2, 5, X_train, labels)
    accuracieslist.append(acc)
def ANNcrossvalidation(X_train,Y_train,k):
    scores=[]
    for train,test in StratifiedKFold(k).split(X_train,Y_train):
        model = Sequential(
        [
            Dense(20, activation="relu"),
            Dropout(rate=0.1),

            Dense(50,activation='relu'),
          
            Dense(1,activation="sigmoid")
        ]
    )
        model.compile(optimizer="RMSPROP",loss="binary_crossentropy",metrics=['accuracy'])
        model.fit(X_train[train],labels[train],batch_size=len(X_train[train]),epochs=200)
        y_predictions=np.round(model.predict(X_train[test]))
        accuracy=accuracy_score(y_predictions, Y_train[test])
        scores.append(accuracy)
        
    return model,scores

des,bovw_features=em.extractfeaturesfromblocks(cropedlist_train, 'sift')
minmax=ps.MinMaxScaler().fit(X_train2)
X_train=minmax.transform(X_train2)
score=fisher_score.fisher_score(X_train2, labels)
idx=fisher_score.feature_ranking(score)
allscores=[]
for i in range (5,95,5):
    X_trainfs=X_train2[:,idx[0:int(len(idx)*i/100)]]
    model,scores=ANNcrossvalidation(X_trainfs, labels, 5)
    allscores.append(scores)
np.mean(allscores,1)    
y_prediction =model.predict(X_test_new)
y_prediction=np.round(y_prediction)
print(confusion_matrix(val_labels,y_prediction))
print(classification_report(val_labels, y_prediction))

def create_ANN_Model(layers,optimizer='adam'):
    model=Sequential()
        # [
        # Dense(200,input_dim=X_train_new.shape[1]),
        # Activation('relu'),
        # Dense(1),
        # Activation("sigmoid")
        # ]
        # )
    for i,nodes in enumerate(layers):
        model.add(Dense(nodes))
        model.add(Activation('relu'))
        model.add(Dropout(0.1))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))    

    model.compile(optimizer=optimizer,loss='binary_crossentropy',metrics=['accuracy'] )
    return model    

model=KerasClassifier(build_fn=create_ANN_Model,verbose=0)
layers=[(20,),(20,50),(50,200,300)]
param_grid=dict(layers=layers,batch_size=[32,64,128],epochs=[50,100,200])
grid=GridSearchCV(model,param_grid,cv=StratifiedKFold(10).split(labels),n_jobs=-1)
grid.fit(X_train_new,labels)
y_prediction =grid.predict(X_test_new)
y_prediction=np.round(y_prediction)
print(confusion_matrix(val_labels,y_prediction))
print(classification_report(val_labels, y_prediction))

model=create_ANN_Model((20,50),'RMSPROP')
model.fit(X_train_new,labels,batch_size=128,epochs=200)
y_prediction =model.predict(X_train2)
y_prediction=np.round(y_prediction)
print(confusion_matrix(val_labels,y_prediction))
print(classification_report(val_labels, y_prediction))

accuracieslist=[]
for k in range(5,91,5):
    fs=SelectPercentile(score_func=f_classif,percentile=k).fit(X_train2,labels)
    X_train=fs.transform(X_train2)
    score=  ANNcrossvalidation(X_train,labels,5)
    accuracieslist.append(np.mean(score))
    
fs=SelectPercentile(score_func=f_classif,percentile=10).fit(X_train2,labels)
X_train=fs.transform(X_train2)
model=create_ANN_Model((20,50),'nadam')
X_test=fs.transform(x_test2)
y_prediction =model.predict(x_test2)
y_prediction=np.round(y_prediction)
print(confusion_matrix(val_labels,y_prediction))
print(classification_report(val_labels, y_prediction))
    