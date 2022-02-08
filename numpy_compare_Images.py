#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 10:44:14 2022

@author: mokegg
"""

# =============================================================================
#import image_similarity_measures
#from image_similarity_measures.quality_metrics import rmse, ssim, sre
from sewar.full_ref import mse, rmse, psnr, uqi, ssim, ergas, scc, rase, sam, msssim, vifp
# import numpy as np
# from keras.preprocessing.image import load_img
# import warnings
# from keras.preprocessing.image import img_to_array
# from keras.preprocessing.image import array_to_img
# =============================================================================
import matplotlib.pyplot as plt    
import cv2
#import tensorflow as tf
# =============================================================================
# img_ano = tf.image.decode_image(tf.io.read_file('Data/train_set/07/16_09_21_07_054.png'))
# img1 = tf.image.adjust_contrast(img_ano, 0.20)
# =============================================================================

img_ano = cv2.imread('Data/train_set/ano/17_11_21_anomalies_033.png',0)
img_ano = cv2.imread('Data/test_set/ano/17_11_21_anomalies_049.png', 0)
img_ano = cv2.imread('Data/train_set/07/16_09_21_07_039.png',0)
img_ano = cv2.imread('Data/test_set/05/17_11_21_05_055.png',0)


import pandas as pd
import os
def compare_img(img_ano):
    rsmes = []
    ssims = []
    #msssims = []
    path = 'Data/templates/'
    for filename in os.listdir(path): #, topdown = True):
        f = os.path.join(path, filename)
        if f.endswith('.png'):
            # print(f)
            img_template = cv2.imread(f,0) #Load the image
            
            # RSME
            rsme = rmse(img_ano.flatten(), img_template.flatten())
            rsmes.append([f, rsme])
            
            #SSIM
            score, diff = ssim(img_ano, img_template)
            diff = (diff * 255).astype("uint8")
            ssims.append([f, score, diff]) 
            
            #MS_SSIM
            #ms_ssim = msssim(img_ano, img_template)
            #msssims.append([f, ms_ssim]) 
            
    # compare based on RSME (root mean square error)        
    df_rsme = pd.DataFrame(rsmes, columns = ['fname', 'rsme'])
    Fname = df_rsme.loc[df_rsme['rsme'].idxmin()].fname
    rsme1 = df_rsme.loc[df_rsme['rsme'].idxmin()].rsme/100
    
    # compare cased on SSIM (structuralsimilarity index)
    df_ssim = pd.DataFrame(ssims, columns = ['fname', 'score', diff])
    Fname1 = df_ssim.loc[df_ssim['score'].idxmax()].fname
    score1 = df_ssim.loc[df_ssim['score'].idxmax()].score
        
    return df_rsme, df_ssim, Fname, Fname1, rsme1, score1

 #('Data/train_set/00/16_09_21_00_005.png')
img_template = cv2.imread('Data/templates/template_07.png')

#df_msssim = compare_img(img_ano)
df_rsme, df_ssim, Fname, Fname1, rsme1, score1 = compare_img(img_ano)

# =============================================================================
# if rsme1 <= 0.25:
#    print(f'RSME = {rsme1} : Good')
# else:
#     print(f'RMSE = {rsme1} : Anomalous')
# 
# pred_image = cv2.imread(Fname)
# print('Fname: {Fname}')
# =============================================================================

if score1 >= 0.75:
   print(f'SSIM ={score1} : Good')
else:
    print(f'SSIM ={score1} : Anomalous')
       
pred_image2 = cv2.imread(Fname1)


# =============================================================================
# plt.figure()
# =============================================================================

#subplot(r,c) provide the no. of rows and columns
f, axarr = plt.subplots(2,1) 

# use the created array to output your multiple images. In this case I have stacked 4 images vertically
axarr[0].imshow(img_ano)
axarr[1].imshow(pred_image2)
#axarr[2].imshow(img_template1)

#####################################################################################################














# =============================================================================
# # Apply compare_image() to folders save results to csv
# from os.path import splitext
# path = 'Data/test_set/ano/'
# result = []
# for filename in os.listdir(path):
#     if filename.endswith('.png'):
#         file_name, ext = splitext(filename)
#         img = cv2.imread(os.path.join(path, filename))
#         Fname_rsme, Fname_ssim, score_rsme, score_ssim = compare_img(img) #compare image with templates
#         result.append([file_name, Fname_rsme, Fname_ssim, score_rsme, score_ssim])
#         
# result = pd.DataFrame(result, columns = ['file_name', 'Fname_rsme', 'Fname_ssim', 'score_rsme', 'score_ssim'])
# 
# from pathlib import Path  
# filepath = Path(path +'csv_result.csv')  
# filepath.parent.mkdir(parents=True, exist_ok=True)  
# result.to_csv(filepath)
# result.loc[(result['score_rsme'] > 0.2) & (result['score_ssim'] < 0.8), 'prediction'] = 1 
# result.loc[(result['score_rsme'] <= 0.2) & (result['score_ssim'] >= 0.8), 'prediction'] = 0
# result.to_csv(filepath)
# =============================================================================
