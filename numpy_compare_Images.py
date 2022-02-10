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
import numpy as np
# =============================================================================
class Compare_images(object):
        
    # input files to check the code
    global img_ano
    _msssin_treshold = 0.79
    ''' The following lines reduce the glare on the images'''

    def __init__(self, img):
        self.img_ano = img
        self.img_ano = self.normalize_img(self.img_ano)
        
        self.img_ano = self.reduce_glare(self.img_ano)
        self.df_ssim, self.Fname1, self.score1, self.msssim_img = self.compare_img(self.img_ano)

        
        #plot array
        self.pred_image2 = cv2.imread(self.Fname1,0) # Predicted image
        #self.img_ano = cv2.cvtColor(self.img_ano ,cv2.COLOR_GRAY2RGB)
        self.msssim_img=msssim(self.img_ano, self.pred_image2, weights=[0.0448, 0.2856, 0.3001, 0.2363, 0.1333], ws=11, K1=0.01, K2=0.03, MAX=None).real
        
        self.diff = self.pred_image2 - self.img_ano

        

    def reduce_glare(self,img):
        blurred = cv2.GaussianBlur(img, (7,7),0)
        thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)[1]
        dst_TELEA = cv2.inpaint(img,thresh,3,cv2.INPAINT_TELEA)
        img_ano = dst_TELEA
        return img_ano



    ''' This function takes in an image and compares it to templates to find a similar image; returns the the most 
    similar template and the score of similarity based on SSIM.'''

    def compare_img(self,img_ano):

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

    

    def get_msssim(self):
        
        if self.msssim_img >= self.msssin_treshold:
            print(f'MSSSIM: multi-scale structural similarity index = { self.msssim_img.real} >= { self.msssin_treshold} : Normal')
            return 'Normal'
        else:
            print(f'MSSSIM: multi-scale structural similarity index = { self.msssim_img.real} < {self.msssin_treshold}: Anomalous')
            return 'Abnormal'
        
    


    def normalize_img(self,img):
        img = img*255.0
        img = np.uint8(img)
        #img = img[:,:,0]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

    

    """ if msssim_img >= msssin_treshold:
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

    msssin_treshold = 0.79"""

    