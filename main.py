from pre import Training_data
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from numpy_compare_Images import Compare_images

from prediction_cnn import Prediction_cnn


def main():
        #custom loss function
    def SSIMLoss(y_true,y_pred):
        return 1-tf.reduce_mean(tf.image.ssim(y_true,y_pred,1.0))

    #autoencoder model
    autoencoder = keras.models.load_model('autoencoder', custom_objects={'SSIMLoss':SSIMLoss})

    #variables for prediction class
    path_model = './Saved models/vgg16_last.h5'
    cnn_model = keras.models.load_model(path_model)

    #make object
    file_handler = Training_data('test', 11)
    test_img = np.load('test_data.npy')

    #load the test data
    test_img = np.load('test_data.npy')
    ano = file_handler.get_ano()
    test_img = np.concatenate((test_img,ano))
    _input = input('random or file: ')
    if _input == 'random':
        samples = file_handler.shuffle_array(test_img)
        amount = input('how many random dices? ')
        samples = samples[:int(amount)]
          
        for sample in samples:
        
            compare = Compare_images(sample)
            prediction = Prediction_cnn(sample,cnn_model)
            predict_dice = autoencoder.predict(np.expand_dims(sample,axis=0))
            #ssim score auto encoder
            ssim1 = tf.image.ssim(predict_dice[0],sample,max_val=1.0,filter_size=1, filter_sigma=1.5, k1=0.01, k2=0.03)
            s = tf.get_static_value(ssim1, partial=False)
            s1 = round(s.real,2)
            s2 = round(compare.msssim_img,2)
            #prediction cnn model
            p_string = prediction.get_prediction_cnn()
            p_value = round(prediction.get_propability(),2)

            #numpy prediction
            if compare.msssim_img >= compare._msssin_treshold:
                Main_title = f'MS_SSIM = {s2} >= { compare._msssin_treshold} : NORMAL'
            else:
                Main_title = f'MS_SSIM  = {s2} < {compare._msssin_treshold}: ABNORMAL'



            if s1 >= compare._msssin_treshold:
                Main_title_auto = f'MS_SSIM = {s1} >= { compare._msssin_treshold} : NORMAL'
            else:
                Main_title_auto = f'MS_SSIM  = {s1} < {compare._msssin_treshold}: ABNORMAL'

            fig, ax = plt.subplots(ncols=3, sharex=True, figsize=(20,20)) #plt.subplots(3,1) 
            
            ax[0].imshow(compare.pred_image2)
            ax[0].set_title(f'NUMPY MS_SSIM: {str(s)}')
            ax[0].text(20,145,Main_title,style='italic',
                bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 10})
            ax[1].imshow(predict_dice[0])
            ax[1].set_title(f'AUTOEN. MS_SSIM: {str(s2)}')
            ax[1].text(20,145,Main_title_auto ,style='italic',
                bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 10})
            ax[2].imshow(sample)
            ax[2].set_title('REAL: ')
            ax[2].text(20,145,str(p_value),style='italic',
                bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 10})
            ax[2].text(100,145,p_string,style='italic',
                bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 10})
        
            plt.show()
    
    
    
    elif _input == 'file':
        file_name = input('1 filename: ')
        samples = file_handler.img_to_array(file_name)
        for sample in samples:
        
            compare = Compare_images(sample)
            prediction = Prediction_cnn(sample,cnn_model)
            predict_dice = autoencoder.predict(np.expand_dims(sample,axis=0))
            #ssim score auto encoder
            ssim1 = tf.image.ssim(predict_dice[0],sample,max_val=1.0,filter_size=1, filter_sigma=1.5, k1=0.01, k2=0.03)
            s = tf.get_static_value(ssim1, partial=False)
            s1 = round(s.real,2)
            s2 = round(compare.msssim_img,2)
            #prediction cnn model
            p_string = prediction.get_prediction_cnn()
            p_value = round(prediction.get_propability(),2)

            #numpy prediction
            if compare.msssim_img >= compare._msssin_treshold:
                Main_title = f'MS_SSIM = {s2} >= { compare._msssin_treshold} : NORMAL'
            else:
                Main_title = f'MS_SSIM  = {s2} < {compare._msssin_treshold}: ABNORMAL'



            if s1 >= compare._msssin_treshold:
                Main_title_auto = f'MS_SSIM = {s1} >= { compare._msssin_treshold} : NORMAL'
            else:
                Main_title_auto = f'MS_SSIM  = {s1} < {compare._msssin_treshold}: ABNORMAL'

            fig, ax = plt.subplots(ncols=3, sharex=True, figsize=(20,20)) #plt.subplots(3,1) 
            
            ax[0].imshow(compare.pred_image2)
            ax[0].set_title(f'NUMPY MS_SSIM: {str(s)}')
            ax[0].text(20,145,Main_title,style='italic',
                bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 10})
            ax[1].imshow(predict_dice[0])
            ax[1].set_title(f'AUTOEN. MS_SSIM: {str(s2)}')
            ax[1].text(20,145,Main_title_auto ,style='italic',
                bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 10})
            ax[2].imshow(sample)
            ax[2].set_title('REAL: ')
            ax[2].text(20,145,str(p_value),style='italic',
                bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 10})
            ax[2].text(100,145,p_string,style='italic',
                bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 10})
        
            plt.show()

        

  


if __name__ == '__main__':
    main()