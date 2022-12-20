import numpy as np
from scipy import stats as st
import tkinter as tk
from tkinter import filedialog
import pickle
import os
import glob
import utility
import cv2 as cv
from process import preprocess, appply_filter

def predict(model, processed_images):
    pred=model.predict(processed_images)
    return (st.mode(pred,keepdims=True)[0][0])

def predict_few(model,images_folder,kernels,lables):
    res=[]
    images_paths=glob.glob(f'{images_folder}/*.jpg')
    for image in images_paths:
        im=cv.imread(os.path.relpath(image))
        if im is None:
            continue
        preprocessed_image=preprocess(im)
        dfs,_=appply_filter(preprocessed_image,kernels,lables)
        X=dfs.to_numpy().T[1:]
        pred=predict(model,X)
        res.append([image,pred])
    return res

def main_select():
    root = tk.Tk()
    root.withdraw()
    model_path = filedialog.askopenfilename(filetypes=[("models", ".model")],title="Вибір моделі",
                                           initialdir = "./models")
    with open(model_path,'rb') as model_file:
        model=pickle.load(model_file)
    model_dir_and_name, _ = os.path.splitext(model_path)
    with open(model_dir_and_name+'.kernels','rb') as model_file:
        kernels=pickle.load(model_file)
    with open(model_dir_and_name+'.lables','rb') as model_file:
        lables=pickle.load(model_file)
    images_folder=utility.open_filedialog('вибір папки зі зорбраженнями')
    for result in predict_few(model,images_folder,kernels,lables):
        print(result)
    
    

if __name__ =="__main__":
    main_select()
