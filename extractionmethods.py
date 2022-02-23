# -*- coding: utf-8 -*-
"""
Created on Sat Apr 10 13:30:19 2021

@author: Djedouani
"""
from skimage.feature import local_binary_pattern,greycomatrix, greycoprops
from sklearn.cluster import KMeans
import cv2 as cv
import numpy as np
import pcanet
def kmeans_clustering(deslist,k):
    kmeans = KMeans(k)
    fvs=np.concatenate((deslist))
    kmeans.fit(fvs)
    bovw_features=[]
    for des in deslist :
        his, y = np.histogram(kmeans.predict(des), bins=k)
        bovw_features.append(his)
    bovw_features=np.array(bovw_features)    
    return bovw_features
    
def extractOrbdes(img,nf):
    orb=cv.ORB_create(nf)
    kp, des = orb.detectAndCompute(img, None)
    return des


def extractsiftdes(img,nf):
    sift = cv.SIFT_create(nfeatures=nf)
    kp, des = sift.detectAndCompute(img, None)
    return des


def extractbovwfeaturesfromblocks(method,cropedimgs,nf,k):
    deslist=[]
    for cropedimg in cropedimgs:
        desim=[]
        for block in cropedimg:
            if method=="sift":
                des=extractsiftdes(block,nf)
            elif method=="orb":   
                des=extractOrbdes(block, nf)
            desim.append(des)
        desim=np.vstack((desim))
        deslist.append(desim)
    return deslist,kmeans_clustering(deslist,k)

def lbp(img,r,p):
    lbparray = local_binary_pattern(img,p,r,'uniform')
    lbphist=np.histogram(lbparray.flatten(),bins=int(lbparray.max()+1))[0]   
    return lbphist

def glcm_features(img,distances,angles):
    glcm=greycomatrix(img,distances,angles,levels=256, normed=True, symmetric=True)
    glcm_contrast=greycoprops(glcm,'contrast').flatten()
    glcm_correlation=greycoprops(glcm,'correlation').flatten()
    glcm_homogeneity=greycoprops(glcm,'homogeneity').flatten()
    glcm_energy=greycoprops(glcm,'energy').flatten()
    glcm_features=np.concatenate((glcm_contrast,glcm_correlation,glcm_homogeneity,glcm_energy))
   
    return glcm_features



def normedf(im_features):
    return im_features/np.max(im_features)  

def extractfeaturesfromblocks(cropedimgs,method):
   if method=="sift" or method=="orb":
        return extractbovwfeaturesfromblocks(method,cropedimgs,25,300)
   fv=[]
   fvs=[]
   for cropedimg in cropedimgs:
      concatenatedfv=[]

      for block in cropedimg:
           if method=="lbp" or method=="glcm":
               if method=="lbp":
                  f1=lbp(block,1,8)
                  f2=lbp(block,2,16)
                  f3=lbp(block,3,24)
                  fv=np.concatenate((f1,f2,f3))
               elif method=="glcm":
                  fv= glcm_features(block,[1,3,5],[0,np.pi/4,np.pi/2])         
               concatenatedfv.append(fv)
      concatenatedfv=np.concatenate(concatenatedfv)     
      fvs.append(concatenatedfv)
   fvs=np.array(fvs)
   return fvs   

def pcanetfun(X_train,X_test):
        pcanetmdl= pcanet.PCANet(
        image_shape=400,
        filter_shape_l1=2, step_shape_l1=1, n_l1_output=3,  # parameters for the 1st layer
        filter_shape_l2=2, step_shape_l2=1, n_l2_output=3,  # parameters for the 2nd layer
        filter_shape_pooling=2, step_shape_pooling=2        # parameters for the pooling layer
    )
        pcanetmdl.validate_structure()
        pcanetmdl.fit(np.array(X_train).astype('float'))  # Train PCANet
        X_train = pcanetmdl.transform(np.array(X_train).astype('float'))
        x_test =pcanetmdl.transform(np.array(X_test).astype('float'))   
        return X_train,x_test
             