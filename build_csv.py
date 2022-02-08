import cv2
import os
import pandas as pd
from numpy_compare_Images import compare_img


#from os.path import splitext

root = 'Data/test_set/00/'
result = []
         
for path, subdirs, files in os.walk(root):
    for file_name in files:
        if file_name.endswith(('.png', '.jpg')):
            
            img1 = cv2.imread(os.path.join(path, file_name),0)
            ''' The following lines reduce the glare on the images'''
    
            blurred = cv2.GaussianBlur(img1, (5,5), 0)
            thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)[1]
            dst_TELEA = cv2.inpaint(img1,thresh,3,cv2.INPAINT_TELEA)
            img1 = dst_TELEA
            
            df_ssim, Fname1, score1, msssim_img = compare_img(img1) 
            print(file_name, Fname1)#compare image with templates
            result.append([file_name, Fname1, score1, msssim_img])
            
    # =============================================================================
results = pd.DataFrame(result, columns = ['file_name', 'template', 'score_ssim', 'msssim'])
results.to_csv('Data/test_set/csv_all_test_ano.csv', index=False)    
print(results['msssim'].min())

