from pre import Training_data
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from numpy_compare_Images import Compare_images
import cv2

class Prediction_cnn(object):

    def __init__(self,img, cnn_model):
        try:
            self.cnn_model=cnn_model
            #convert to 3 channels
            
            channel_three = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
            channel_three = np.expand_dims(channel_three,axis=0)    
            #predict model
            self.predict_dice_cnn = self.cnn_model.predict(channel_three)
            self.prediction = np.argmax(self.predict_dice_cnn)
            #get highest propa
        except:
            print('wrong path, error somewhere')
        

    def get_prediction_cnn(self)->float:
        #return 
        if self.prediction == 1:
            return 'NORMAL'
        else:
            
            return 'ABNORMAL'   

    def get_propability(self):
        propa = self.predict_dice_cnn[0][self.prediction] *100
        return propa
