import numpy as np
from scipy.linalg import norm
import cv2 as cv
from process import preprocess,appply_filter
import os
import pandas as pd

def prepare_for_scores(image_names,kernels,lables):
    X=[]
    Y=[]
    for file,cls in zip(image_names['file'],image_names['class']):
        try:
            path=os.path.relpath(file)
            im=cv.imread(path)
            preprocessed=preprocess(im,size=(32,32))
            dataframe,raw_features=appply_filter(preprocessed,kernels,lables)
            arr=[dataframe.iloc[:,i].to_numpy() for i in range(1,len(kernels)+1)]
            X.append(arr)
            Y.append(cls)
        except Exception as e:
            #print(e)
            continue
    return X,Y

def calculate_score(X,Y,norm_type=np.inf):
    X=np.array(X)
    Y=np.array(Y)
    i=0
    scores=[]
    for k in range(0,X.shape[1]):
        i=i+1
        m1k=X[:,k][Y==0].mean(0)
        m2k=X[:,k][Y==1].mean(0)
        sig1k=X[:,k][Y==0].std(0)
        sig2k=X[:,k][Y==1].std(0)
        j=(norm(m1k-m2k,norm_type)**2)/(norm(sig1k**2,norm_type)+norm(sig2k**2,norm_type))
        scores.append(j)
    return scores

def select_filters(X,Y,all_kernels,all_lables,number_of_filters =30):
    scores=calculate_score(X,Y);
    ScoreBoard=pd.DataFrame(scores,[i for i in range(0,len(all_kernels))])
    ScoreBoard=ScoreBoard.dropna()
    ScoreBoard=ScoreBoard.sort_values(by=0)
    selected_kernels=[all_kernels[i] for i in ScoreBoard.iloc[-number_of_filters:].index]
    selsected_lables=[all_lables[i] for i in ScoreBoard.iloc[-number_of_filters:].index]
    return selected_kernels,selsected_lables
