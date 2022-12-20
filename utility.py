import tkinter as tk
from tkinter import filedialog
import glob
import pandas as pd
import numpy as np
import cv2 as cv

def read_images(sample_files, else_files):
    files=glob.glob(f'{sample_files}/*.jpg')
    data =pd.DataFrame({'file':files,'class':[1]*len(files)})
    other_files=glob.glob(f'{else_files}/*.jpg')
    other_data=pd.DataFrame({'file':other_files,'class':[0]*len(other_files)})
    data=data.append(other_data)
    data = data.sample(frac=1).reset_index(drop=True)
    return data

def open_filedialog(title='вибір файлу'):
    root = tk.Tk()
    root.withdraw()
    directory = filedialog.askdirectory(title=title)
    return directory

def generate_filters(ksize=5,phi=0):
    kernels=[]
    lables=[]
    for theta in range(4):
        theta=theta/4.*np.pi
        for sigma in(1,3,5):
            for lamda in np.arange(0,np.pi,np.pi/4):
                for gamma in(0.05,0.5):
                    label = "Gabor "+" ".join(str(x) for x in[theta,sigma,lamda,gamma])
                    kernel=cv.getGaborKernel((ksize,ksize),sigma,theta,lamda,gamma,phi,ktype=cv.CV_32F)
                    if np.isnan(kernel[0][0]):continue
                    kernels.append(kernel)
                    lables.append(label)
    return kernels,lables