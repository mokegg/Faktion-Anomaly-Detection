# Faktion-Anomaly-Detection 🧩 🎲


***
### Description
This project is about . The goal of the project is to investigate and implement computer vision methods for the purpose of Anomaly detection. The dices dataset used is provided by [Faktion](https://www.faktion.com/). The dataset consits of more than 7000 (128 x 128) images of dices. The dataset are grouped in 11 diefferent die faces. The dataset also includes  morethan 350 anomaly examples. The dataset is divided in to training and testing sets. 

<img width="450" alt="dice" src="https://user-images.githubusercontent.com/46165841/153586153-4a8ef509-e527-4df2-9a8a-9ec231b10abf.png">

<img width="531" alt="image" src="https://user-images.githubusercontent.com/46165841/153586772-7a0314a7-7433-48ab-9eb4-683649d0c2c4.png">

 Three different approaches were followed:
1. **Numpy array method** - Takes in an image and compares it to templates to find a similar Template; returns the the most 
similar template and classifies it normal or abnormal based the score of similarity based on MS_SSIM score.

<img width="290" alt="image" src="https://user-images.githubusercontent.com/46165841/153586864-94351ea2-9eaf-45bf-97b5-4226bd93a57a.png">

2. **Transfer learning** - Convolutional neural networks (CNN) - Takes in an image and uses existing pretrained model `VGG16` to predict  normal or abnormal.
<img width="400" alt="image" src="https://user-images.githubusercontent.com/46165841/153588023-c824ca55-6eab-483d-af51-2dbd4d628329.png">

3. **Convolutional autoencoder** - This model is trained on only the normal training dataset. Given any image, it reconstructs the image and it determines whether the image is Normal or Abnormal based on the similairt of the original image and the reconstructed one.

<img width="400" alt="image" src="https://user-images.githubusercontent.com/46165841/153588169-90ada61f-cf2e-4efb-a0f2-d9042a8b4b9f.png">

***
### Installation
The environment at which the codes are developed is provided in the `requirements.txt` file. To run the code, the necessary libraries should be installed based on that environment.


***
### Usage


The three methods are implemented in a single code named main.py
  

***
### Visuals

Here are sample visuals:

![ab](https://user-images.githubusercontent.com/46165841/153591009-9e9a6014-1e17-497a-8b88-7fbb697902f5.png)

![Figure_1](https://user-images.githubusercontent.com/46165841/153591087-861c122a-b586-41e3-b6c9-c23028d9849e.png)

##### Optimizing the F1_score: 0.973

<p align="center">

***
#### Contributors
This project is worked-out by the following team:
                                                                                  
- [Mekonnen Gebrehiwot](https://github.com/mokegg)                                                                                           
- [Sebastian Chavez](https://github.com/sebastianchavezz) 
- [YusufAkcakaya](https://github.com/yusufakcakaya)      
                                                                                                                                    
***
### What could be improved 
- Measure the accuracy of Autoencoder model
- Improve  Autoencoder model (wrt images reconstructed)
- Reduce overfitting of VGG16
- Increase the output classes of VGG16
- Deployment



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
