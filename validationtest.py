# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 22:34:30 2021

@author: Djedouani
"""

import pickle
import classification
X_test=pickle.load(open('X_test.p','rb'))
blindlabels=pickle.load(open('val_labels.p','rb'))
transformer=pickle.load(open('transformer.p','rb'))
scalertransformer=pickle.load(open('scalertransformer.p','rb'))
model=pickle.load(open('bestmodel.p','rb'))

X_test=transformer.transform(X_test)
X_test=scalertransformer.transform(X_test)


classification.testmodel(model,X_test,blindlabels)
