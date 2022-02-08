#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 10:39:59 2022

@author: mokegg
"""


# Apply compare_image() to folders save results to csv
import cv2
import os
import pandas as pd
from compare_img import compare_img


import tensorflow as tf

from os.path import splitext

path = 'Data/test_set/00/'
result = []
for filename in os.listdir(path):
    if filename.endswith('.png'):
        file_name, ext = splitext(filename)
        
        img_ano = tf.image.decode_image(tf.io.read_file(os.path.join(path, filename)))
       # img_ano = tf.image.rgb_to_grayscale(img_ano)
        img1 = tf.image.adjust_contrast(img_ano, 0.20)
        img1 = tf.image.random_brightness(img1, max_delta = 0.1, seed=None)
        
        #img1 = cv2.imread(os.path.join(path, filename))
        df_ssim, Fname1, score1 = compare_img(img1) 
        print(file_name, Fname1)#compare image with templates
        result.append([file_name, Fname1, score1])
            
    # =============================================================================
results = pd.DataFrame(result, columns = ['file_name', 'template', 'score_ssim'])
results.to_csv('Data/test_set/00/csv.csv', index=False)    
print(results['score_ssim'].max())
# =============================================================================
#result.to_csv(filepath)

# =============================================================================
# filepath = Path('Data/templates/new/result_ano.csv')  
# filepath.parent.mkdir(parents=True, exist_ok=True)  
# result_00.to_csv(filepath) 
# =============================================================================

