import cv2 as cv
import glob
import numpy as np


def extractimgs(classpath,dtype):
    imgs = []
        
    for file in sorted(glob.glob(classpath),key=len):
        if dtype==-1:
            img = cv.imread(file,-1)
        elif dtype==0:
            img = cv.imread(file,0)
        imgs.append(img)    
    return imgs


def clahefun(imgs,clipLimit,tileGridSize):
    clahe=cv.createCLAHE(clipLimit=clipLimit,tileGridSize=tileGridSize)
    nimgs=[]
    for img in imgs:
        nimgs.append(clahe.apply(img))
    return nimgs

def minmaxfun(imgs,levels):
    nimgs=[]
    for img in imgs:
        minmax=((img-np.min(img))/(np.max(img)-np.min(img))*levels).astype(int)
        nimgs.append(minmax)     
    return nimgs    

def readimgsfromfolder(nimgspath): 
    imgspath = sorted(glob.glob(nimgspath),key=len)
    imgsarray = []
    for imgfile in imgspath:
            img = cv.imread(imgfile, -1)
            imgsarray.append(img)
    return imgsarray


def savenimgs(imgsarray, filepath):
    for i in range(len(imgsarray)):
        cv.imwrite(filepath+"\\"+str(i)+".tif", imgsarray[i])
   
def crop(img, windowsize,overlap):
    windows = []
    # for r in range(0, img.shape[0], windowsize):
    #     for c in range(0, img.shape[1], windowsize):
    #         windows.append(img[r:r+windowsize, c:c+windowsize])
    def start_points(size, split_size, overlap):
                points = [0]
                stride = int(split_size * (1-overlap))
                counter = 1
                while True:
                    pt = stride * counter
                    if pt + split_size >= size:
                        points.append(size - split_size)
                        break
                    else:
                        points.append(pt)
                    counter += 1
                return points 
    img_h, img_w = img.shape
    X_points = start_points(img_w, windowsize,overlap)
    Y_points = start_points(img_h, windowsize,overlap)
    for i in Y_points:
        for j in X_points:
            split = img[i:i+windowsize, j:j+windowsize]
            windows.append(split)
    return windows

def getnormeddb(class0,class1):
    normdb = []
    labels=[]
    for i in range(0, len(class0)+len(class1)):
        if i % 2 == 0:
            normdb.append(class0[int(i/2)])
            labels.append(0)
        else:
            normdb.append(class1[int((i-1)/2)])
            labels.append(1)
    labels=np.array(labels)
       
    return normdb,labels

def to8bit(imgs):
    newimgs=[]
    for img in imgs:
        newimgs.append((img/256).astype('uint8'))
    return newimgs     

