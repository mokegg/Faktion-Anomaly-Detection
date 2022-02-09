
import tensorflow as tf
import numpy as np 
import os 
from glob import glob

#handles the images of the files, loops through files and return a list of all the dices
class Preprocesser(object):


    def __init__(self,files:list,classes_amount):
        try:
            if classes_amount ==6:
                #gets all 'raw'dices
                self.dices_raw = self.get_all_dices(files)
                #converts the dataset into 6 classes
                self.dices = self.convert_to_six(self.dices_raw)
        
            elif classes_amount ==11:
                #gets all 'raw'dices
                self.dices = self.get_all_dices(files)
                #converts the dataset into 6 classes
            
            else:
                print("the classes options are 6 or 11")

        except:
            print('Preprocesser file needs 6 or 11 as input for classe')
    
    #gets the images as array from data
    def get_dices_in_files(self,file_array: list):
        data = []
   
        #looping through files array
        for file in file_array:
            #getting the image data
            image = tf.keras.utils.load_img(file, grayscale=False, color_mode='grayscale', target_size=None,interpolation='nearest')
            #converting everything to numpy array
            input_arr = tf.keras.preprocessing.image.img_to_array(image)
            input_arr = np.array([input_arr]) 
            #normalize the array to a float from 0 to 1
            input_arr = input_arr/255
            #add dice image to training data
            data.append(input_arr[0])

        dices_array = np.array(data)
        #print('files: '+type(dices_array))
        return dices_array

    #loops through files and calls get_dices_in_files to get the dices per file 
    def get_all_dices(self,files: list):
        dices = []
        for file in files:
            f = glob(file)
            dices.append(self.get_dices_in_files(f))
        #print('dices: '+type(dices))
        return dices

    #converts the 11 classes to 6 classes
    def convert_to_six(self, dices_raw:list):
        dices = []
        for i in range(7):
            if i == 1:
                temp = np.concatenate([dices_raw[i], dices_raw[i+1]])
                dices.append(temp)
                continue
            if i ==2:
                continue
            if i == 3:
                temp = np.concatenate([dices_raw[i], dices_raw[i+1]])
                dices.append(temp)
                continue
            if i ==4:
                continue
            else:
                dices.append(dices_raw[i])

        t = np.concatenate([dices_raw[7],dices_raw[8]])

        t= np.concatenate([t,dices_raw[9]])
        t= np.concatenate([t,dices_raw[10]])
        dices.append(t)
        
        return dices
   

#Hadles the training data and test data
class Training_data(object):
    #files
    training_files = ['./train_set/00/*','./train_set/01/*','./train_set/02/*','./train_set/03/*','./train_set/04/*','./train_set/05/*','./train_set/06/*','./train_set/07/*','./train_set/08/*','./train_set/09/*','./train_set/10/*'] 
    test_files =  ['./test_set/00/*','./test_set/01/*','./test_set/02/*','./test_set/03/*','./test_set/04/*','./test_set/05/*','./test_set/06/*','./test_set/07/*','./test_set/08/*','./test_set/09/*','./test_set/10/*']

    global anomalies



    def __init__(self,type:str, classes_amount:int):
        """input: training or test; classes: 6 or 11
        """
        try:
            self.get_Data(type,classes_amount)
        except:
            print('something wrong, see if all arguments are filled!')
    
    #check if training data or test data
    def get_Data(self,type:str,classes_amount:int):
         
        if type == 'training':
            training_data = 'training_data.npy'
            training_labels = 'training_labels.npy'
            if os.path.exists(training_data) and os.path.exists(training_labels):
                print('Files already exis, import them!') 
                print(f'data: {training_data}| labels: {training_labels}')
            
            else:
                preprocessing = Preprocesser(self.training_files,classes_amount)
                
                self.raw_dices = preprocessing.dices
                self.labels = self.get_labels(self.raw_dices) 
                self.dices = self.stack_training_data(self.raw_dices)
                anomalies = preprocessing.get_all_dices(['./train_set/ano/*'])
                #print(type(anomalies))
                self.set_ano(anomalies)
                self.save_data(self.dices,training_data)
                self.save_data(self.labels,training_labels)
                print(f'data: {training_data}| labels: {training_labels}')
        
        elif type == 'test':
            test_data = 'test_data.npy'
            test_labels = 'test_labels.npy'
            if os.path.exists(test_data) and os.path.exists(test_labels):
                print('Files already exis, import them!') 
                print(f'data: {test_data}| labels: {test_labels}')
            
            else:  
                preprocessing = Preprocesser(self.test_files,classes_amount)
                self.raw_dices = preprocessing.dices
                self.labels = self.get_labels(self.raw_dices) 
                self.dices = self.stack_training_data(self.raw_dices)
                
                anomalies = preprocessing.get_all_dices(['./test_set/ano/*'])
                
                self.set_ano(anomalies)
                self.save_data(self.dices,test_data)
                self.save_data(self.labels,test_labels)
                print(f'data: {test_data}| labels: {test_labels}')
       
        
        else:
            print(f'nameError: full training, training, test, full test')
    
    #save the data
    def save_data(self, data, name):

        np.save(name,data)
    
    #getting all the labels for the data with respect to the amount of classes
    def get_labels(self,_dices):
        labels = []
        for i in range(len(_dices)):
            for j in range(len(_dices[i])):
                labels.append(i)
        arr = np.array(labels)
        arr = arr.reshape((-1,1))
        return arr

    #stacks the raw data into 1 np array object
    def stack_training_data(self,_dices: list):
        training_data = np.concatenate([_dices[0],_dices[1]])
        for i in range(2,len(_dices)):
            training_data = np.concatenate([training_data,_dices[i]])
        return training_data
    
    #list of dices per category
    def get_raw(self):
        raw = np.array(self.raw_dices, dtype='object')
        return raw

    #flatten the labels to a vector
    def labels_to_1d(self):
        labels = []
        for i in range(len(self.raw_dices)):
            for j in range(len(self.raw_dices[i])):
                labels.append(i)
        arr = np.array(labels)
        
        return arr
    
    #get the anomoly data from object
    def get_ano(self)->object:
        
        return np.array(self.anomalies[0])
    
    #set the anomaly data to object
    def set_ano(self,ano):
        self.anomalies = ano