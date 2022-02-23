# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 16:53:47 2021

@author: Djedouani
"""
import prepross
import pickle
import numpy as np
blindpath = "blind/*"
class0path = "class0/*"
class1path = "class1/*"

#pickle.load(open('db.p','rb'))

blindlabels=np.array([0,1,0,0,0,0,0,0,0,0,1,0,1,1,0,0,
                      1,0,0,1,0,1,1,1,1,0,1,0,1,1,1,0,
                      0,0,1,1,1,1,0,1,1,0,1,0,0,1,0,1,
                      1,1,1,1,1,0,0,1,0,0])

blindimgs=prepross.readimgsfromfolder(blindpath)
class0imgs=prepross.readimgsfromfolder(class0path)
class1imgs=prepross.readimgsfromfolder(class1path)
db,labels=prepross.getnormeddb(class0imgs,class1imgs)


""" #originaldataset
pickle.dump(db,open("db.p",'wb'))
pickle.dump(blindimgs,open("validation.p",'wb'))
pickle.dump(labels,open("labels.p",'wb'))
pickle.dump(blindlabels,open('val_labels.p','wb'))
"""

 #minmax normalization
minmaxnormed8bit=prepross.minmaxfun(prepross.to8bit(db),2**8-1)
minmaxvalnormed8bit=prepross.minmaxfun(prepross.to8bit(blindimgs),2**8-1)

pickle.dump(minmaxnormed8bit,open('normed_db_8bit_minmax.p','wb'))
pickle.dump(minmaxvalnormed8bit,open('normed_val_8bit_minmax.p','wb'))


cliplimit=0.02
gridtilesize=(40,40)

clahenormed16bit=prepross.clahefun(db,cliplimit,gridtilesize)
clahevalnormed16bit=prepross.clahefun(blindimgs,cliplimit,gridtilesize)
clahenormed8bit=prepross.to8bit(clahenormed16bit)
clahevalnormed8bit=prepross.to8bit(clahevalnormed16bit)
pickle.dump(clahenormed8bit,open('normed_db_8bit_clahe.p','wb'))
pickle.dump(clahevalnormed8bit,open('normed_val_8bit_clahe.p','wb'))



