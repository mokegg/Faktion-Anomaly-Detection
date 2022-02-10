#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 10:44:14 2022

@author: mokegg
"""

# =============================================================================
from sewar.full_ref import  ssim, msssim
import matplotlib.pyplot as plt    
import cv2
import pandas as pd
import os
# =============================================================================

# input files to check the code

img_ano = cv2.imread('Data/test_set/ano/17_11_21_anomalies_050.png',0)
#img_ano = cv2.imread('Data/test_set/ano/17_11_21_anomalies_038.png', 0)
#img_ano = cv2.imread('Data/test_set/ano/17_11_21_anomalies_031.png',0)
#img_ano = cv2.imread('Data/test_set/00/17_11_21_00_001.png',0)
#img_ano = cv2.imread('Data/test_set/03/26_01_22_03_236.png',0)
#img_ano = cv2.imread('Data/test_set/ano/17_11_21_anomalies_037.png',0)
#img_ano = cv2.imread('Data/test_set/ano/04.png',0)

''' The following lines reduce the glare on the images'''

blurred = cv2.GaussianBlur(img_ano, (7,7), 0)
thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)[1]
dst_TELEA = cv2.inpaint(img_ano,thresh,3,cv2.INPAINT_TELEA)
img_ano = dst_TELEA



''' This function takes in an image and compares it to templates to find a similar image; returns the the most 
similar template and the score of similarity based on SSIM.'''

def compare_img(img_ano):

    ssims = []

    path = 'Data/templates/'
    for filename in os.listdir(path): #, topdown = True):
        f = os.path.join(path, filename)
        if f.endswith('.png'):
            img_template = cv2.imread(f,0) #Load the image
            
            #CalSSIM
            score, diff = ssim(img_ano, img_template)
            diff = (diff * 255).astype("uint8")
            ssims.append([f, score, diff]) 
                
    # compare cased on SSIM (structuralsimilarity index)
    df_ssim = pd.DataFrame(ssims, columns = ['fname', 'score', diff])
    Fname1 = df_ssim.loc[df_ssim['score'].idxmax()].fname
    score1 = df_ssim.loc[df_ssim['score'].idxmax()].score
    pred_image2 = cv2.imread(Fname1,0)
    msssim_img=msssim(img_ano, pred_image2, weights=[0.0448, 0.2856, 0.3001, 0.2363, 0.1333], ws=11, K1=0.01, K2=0.03, MAX=None).real
    return df_ssim, Fname1, score1, msssim_img

df_ssim, Fname1, score1, msssim_img = compare_img(img_ano)
       
pred_image2 = cv2.imread(Fname1,3) # Predicted image
img_ano = cv2.cvtColor(img_ano ,cv2.COLOR_GRAY2RGB)
diff = pred_image2 - img_ano  # The difference between the two images
#diff = cv2.cvtColor(diff,cv2.COLOR_GRAY2RGB)


# NORMAL OR ANOMALOUS based om MS_SSIM

msssim_img=msssim(img_ano, pred_image2, weights=[0.0448, 0.2856, 0.3001, 0.2363, 0.1333], ws=11, K1=0.01, K2=0.03, MAX=None)

msssin_treshold = 0.79

if msssim_img >= msssin_treshold:
   Main_title = f'MS_SSIM = { round(msssim_img.real, 2)} >= { msssin_treshold} : Normal'
else:
   Main_title = f'MS_SSIM  = {round(msssim_img.real, 2)} < {msssin_treshold}: Anomalous'

fig, ax = plt.subplots(nrows=3, sharex=True, figsize=(3, 5)) #plt.subplots(3,1) 

ax[0].imshow(img_ano)
ax[0].set_title('Input image', x=-1.2, y=0.5)
ax[1].imshow(pred_image2)
ax[1].set_title('Matching Template', x=-1.2, y=0.35)
ax[2].imshow(diff)
ax[2].set_title('Difference', x=-1.2, y=0.25)

fig.suptitle(f'{Main_title}', color="blue",fontsize=16)


#####################################################################################################

# NORMAL OR ANOMALOUS based om MS_SSIM

msssim_img=msssim(img_ano, pred_image2, weights=[0.0448, 0.2856, 0.3001, 0.2363, 0.1333], ws=11, K1=0.01, K2=0.03, MAX=None)

msssin_treshold = 0.79

if msssim_img >= msssin_treshold:
   print(f'MSSSIM: multi-scale structural similarity index = { msssim_img.real} >= { msssin_treshold} : Normal')
else:
    print(f'MSSSIM: multi-scale structural similarity index = { msssim_img.real} < {msssin_treshold}: Anomalous')
 
    
