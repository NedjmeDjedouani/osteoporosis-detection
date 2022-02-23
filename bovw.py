# -*- coding: utf-8 -*-

import numpy as np
import cv2  as cv
import glob
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import svm

n_clusters=10
n_kp=100
pathsrc="images/*"

def SIFTdetectandcompute(imgspath,n_kp):
    path=glob.glob(imgspath)
    sift=cv.SIFT_create(n_kp)
    descriptorlist=[]
    for file in path:
        img=cv.imread(file)
        img = cv.normalize(img, None, 0, 255, cv.NORM_MINMAX).astype('uint8')
        gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        kp,des=sift.detectAndCompute(gray,None)
        descriptorlist.append(des)
    return descriptorlist

descriptor_list=SIFTdetectandcompute(pathsrc, n_kp)    
# bovw_features=des[1]
# plt.show
kmean=KMeans(n_clusters)
def extracthistograms(descriptor_list):
    his=[]
    for descriptorvector in descriptor_list:
        kmean.fit(descriptorvector)
        his.append(np.histogram(kmean.predict(descriptorvector),n_clusters)[0])
    return his

histograms=np.array(extracthistograms(descriptor_list))
clf=svm.SVC()
clf.fit(np.array(histograms),[0,1])


# def build_histogram(descriptor_list, cluster_alg):
#     histogram = np.zeros(len(cluster_alg.cluster_centers_))
#     cluster_result =  cluster_alg.predict(descriptor_list)
#     print(cluster_result)
#     for i in cluster_result:
#         histogram[i] += 1.0
      
#     return histogram

# his=build_histogram(des[1],kmean)
# his2=np.histogram(kmean.predict(des[1]),4)
# m=kmean.predict(bovw_features)

# plt.hist(m,n_clusters)

