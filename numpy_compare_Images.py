#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 10:44:14 2022

@author: mokegg
"""

# =============================================================================
# import image_similarity_measures
# from image_similarity_measures.quality_metrics import rmse, ssim, sre
from sewar.full_ref import mse, rmse, psnr, uqi, ssim, ergas, scc, rase, sam, msssim, vifp
# import numpy as np
# from keras.preprocessing.image import load_img
# import warnings
# from keras.preprocessing.image import img_to_array
# from keras.preprocessing.image import array_to_img
# =============================================================================
import matplotlib.pyplot as plt    
import cv2

# =============================================================================
# import scipy.spatial.distance as dist
# =============================================================================

img_ano = cv2.imread('Data/train_set/ano/17_11_21_anomalies_033.png')


import os
import pandas as pd

def compare_img(img_ano):
    rsmes = []
    ssims = []
    path = 'Data/templates/'
    for filename in os.listdir(path): #, topdown = True):
        f = os.path.join(path, filename)
        if f.endswith('.png'):
            # print(f)
            img_template = cv2.imread(f) #Load the image
            rsme = rmse(img_ano.flatten(), img_template.flatten())
            rsmes.append([f, rsme])
            
            
            score, diff = ssim(img_ano, img_template)
            diff = (diff * 255).astype("uint8")
            ssims.append([f, score, diff]) 
            
    # compare based on RSME (root mean square error)        
    df_rsme = pd.DataFrame(rsmes, columns = ['fname', 'rsme'])
    Fname = df_rsme.loc[df_rsme['rsme'].idxmin()].fname
    rsme1 = df_rsme.loc[df_rsme['rsme'].idxmin()].rsme/100
    
    # compare cased on SSIM (structuralsimilarity index)
    df_ssim = pd.DataFrame(ssims, columns = ['fname', 'score', diff])
    Fname1 = df_ssim.loc[df_ssim['score'].idxmax()].fname
    score1 = df_ssim.loc[df_ssim['score'].idxmax()].score
    return Fname, Fname1, rsme1, score1

img_ano = cv2.imread('Data/train_set/05/16_09_21_05_063.png')

Fname, Fname1, rsme1, score1 = compare_img(img_ano)

if rsme1 <= 0.3:
   print(f'RSME = {rsme1} : Good')
else:
    print(f'RMSE = {rsme1} : Anomalous')

pred_image = cv2.imread(Fname)


if score1 >= 0.75:
   print(f'SSIM ={score1} : Good')
else:
    print(f'SSIM ={score1} : Anomalous')
       
pred_image2 = cv2.imread(Fname1)


# =============================================================================
# plt.figure()
# =============================================================================

#subplot(r,c) provide the no. of rows and columns
f, axarr = plt.subplots(3,1) 

# use the created array to output your multiple images. In this case I have stacked 4 images vertically
axarr[0].imshow(img_ano)
axarr[1].imshow(pred_image)
axarr[2].imshow(pred_image2)


# Apply compare_image() to folders save results to csv
from os.path import splitext
path = 'Data/test_set/ano/'
result = []
for filename in os.listdir(path):
    if filename.endswith('.png'):
        file_name, ext = splitext(filename)
        img = cv2.imread(os.path.join(path, filename))
        Fname_rsme, Fname_ssim, score_rsme, score_ssim = compare_img(img) #compare image with templates
        result.append([file_name, Fname_rsme, Fname_ssim, score_rsme, score_ssim])
        
result = pd.DataFrame(result, columns = ['file_name', 'Fname_rsme', 'Fname_ssim', 'score_rsme', 'score_ssim'])

from pathlib import Path  
filepath = Path(path +'csv_result.csv')  
filepath.parent.mkdir(parents=True, exist_ok=True)  
result.to_csv(filepath)
result.loc[(result['score_rsme'] > 0.2) & (result['score_ssim'] < 0.8), 'prediction'] = 1 
result.loc[(result['score_rsme'] <= 0.2) | (result['score_ssim'] >= 0.8), 'prediction'] = 0
result.to_csv(filepath)