# Faktion-Anomaly-Detection


***
### Description
This project is about . The goal of the project is to investigate and implement computer vision methods for the purpose of Anomaly detection. The dices dataset used is provided by [Faktion](https://www.faktion.com/). The dataset consits of morethan 7000 (128 x 128) images of dices. The dataset are grouped in 11 diefferent die faces. The dataset also includes  morethan 350 anomaly examples. The dataset is divided in to training and testing sets. 

 Three different approaches were followed:
1. Numpy array method - Takes in an image and compares it to templates to find a similar Template; returns the the most 
similar template and classifies it normal or abnormal based the score of similarity based on MS_SSIM score 
2. Transfer learning - Convolutional neural networks (CNN) - Takes in an image and uses existing pretrained model `VGG16` to predict  normal or abnormal
3. Convolutional autoencoder - This model is trained on only the normal training dataset. Given any image, it reconstructs the image and it determines whether the image is Normal or Abnormal based on the similairt of the original image and the reconstructed one

***
### Installation
The environment at which the codes are developed is provided in the `requirements.txt` file. To run the code, the necessary libraries should be installed based on that environment.


***
### Usage


The three methods are implemented in a single code named pipeline.py
  

***
### Visuals

Here are sample visuals:

##### Optimizing the F1_score:
<p align="center">
  <img src="<p align="left">
  <img src="![Screenshot from 2021-10-13 15-32-44](Readme_pics/sample0.png)" width="400" height="90" /> </p>

Sample output from the numpy approach.
<p align="center">
  <img src="<p align="left">
  <img src="![![dataScaprePandas](Readme_pics/sample1.png))" width="800" height="500" /> </p>

***
#### Contributors
This project is worked-out by the following team:
                                                                                  
- [Mekonnen Gebrehiwot](https://github.com/mokegg)                                                                                           
- [Sebastian Chavez](https://github.com/sebastianchavezz) 
- [YusufAkcakaya](https://github.com/yusufakcakaya)      
                                                                                                                                    
***
### What could be improved 


***
### Timeline
Feb 2021

***
### Personal Situation
This was a group project at [BeCode](https://becode.org/). The challange given by [Faktion](https://www.faktion.com/)

Here is how you can contact us :
##### LinkedIn :                                                                           
- [Mekonnen Gebrehiwot](https://www.linkedin.com/in/mekonnen1/?originalSubdomain=be)
- [Sebastian Chavez](https://www.linkedin.com/in/sebastian-chavez-2-9a0790186/) 
- [YusufAkcakaya](https://www.linkedin.com/in/yusuf-ak%C3%A7akaya-9526a0171/)
                                                                                                                                           
##### Email :                                                                                                                                          
- mekonnen.gebrehiwot1@gmail.com
- sebastianchavez940@gmail.com 
- yusufakcakaya14@gmail.com                                                                                                                                    
