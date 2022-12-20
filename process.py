import cv2 as cv
import pandas as pd
import os
import numpy as np
from utility import generate_filters

def preprocess(img,size=(32,32)):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    resized=cv.resize(gray,size)
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    normalized = clahe.apply(resized)
    return normalized
    
def appply_filter(img,kernels,lables):
    df=pd.DataFrame()
    df['Original Image']=img.reshape(-1)
    raw_features=[]
    for kernel,label in zip(kernels,lables):
        fimg = cv.filter2D(img,cv.CV_8UC3,kernel)
        raw_features.append(fimg)
        df[label]=fimg.reshape(-1)
    return df,raw_features 

def form_data(frame,kernels=None,lables=None,size=(32,32)):
    if kernels==None or lables==None:
        kernels,lables=generate_filters()
    X=[]
    Y=[]
    for i in range(frame.shape[0]):
        path=os.path.relpath(frame.iloc[i]['file'])
        im=cv.imread(path)
        try:
            len(im)
        except:
            print("не можу відкрити: "+path)
            continue
        cls=[int(frame.iloc[i]['class'])]
        proc_im=preprocess(im,size)
        data_f,raw_features=appply_filter(proc_im,kernels,lables)
        feat_array=np.array(data_f.to_numpy().T)
        X.append(feat_array[1:])
        Y.append(cls*(data_f.shape[1]-1))
    X=np.array(X)
    Y=np.array(Y)
    X=X.reshape(-1,X.shape[-1])
    Y=Y.reshape(-1)
    return X,Y