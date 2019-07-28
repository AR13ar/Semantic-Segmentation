# -*- coding: utf-8 -*-
"""
@author: Aditya Raj
"""
import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, GaussianNoise, Flatten, Dropout, Lambda, Add
from keras.models import Model, Sequential
from keras.losses import binary_crossentropy, mae, mse
from keras.optimizers import adadelta, RMSprop, Adam, SGD
from keras.callbacks import TensorBoard
from keras.regularizers import l2
from keras.layers.advanced_activations import LeakyReLU
from keras.models import load_model
import tensorflow as tf
from keras import layers
from numpy import *
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split 
import random
from keras.layers import merge
from PIL import Image
from Autoencoder_model import enco, deco
from Data_generator import MY_data
image_x = 320
image_y = 320

path_train = "D:\\PROJECTS\\data\\raw\\" #Path of training samples
list_train = os.listdir(path_train)

# This was used to save the augmented files in the same folder as the training samples
#for i in list_train:
#    img = Image.open(path_train + i).convert("RGB")
#    img = np.array(img)
#    img_rot = img.rotate(90, PIL.Image.NEAREST)
#    cv2.imwrite(path_train + 'rot_' + i  , np.array(img_rot))
 


path_gt = "D:\\PROJECTS\\data\\gt\\" #Path of ground truth
list_gt = os.listdir(path_gt)
# This was used to save the augmented files in the same folder as the GT samples
#for i in list_test:
#    img = Image.open(path_gt + i).convert("RGB")
##    img = np.array(img)
#    img_rot = img.rotate(90, PIL.Image.NEAREST)
#    cv2.imwrite(path_gt + 'rot_' + i  , np.array(img_rot))

path_test = "D:\\PROJECTS\\test_raw\\"
list_test = os.listdir(path_test)

input_shape = (image_x, image_y,1) #Input For Encoder
input_img = Input(shape= input_shape)
size_ls = (40,40,256)   #Latent Space
size_input = Input(size_ls)
   
out_en = enco(input_img)
enco_model = Model(inputs = input_img, outputs = out_en) #Defining Encoder Model
enco_model.summary()

out_decoder = deco(size_ls)
out_de = out_decoder(size_input)
deco_model = Model(inputs = size_input, outputs = out_de) #Defining Decoder Model
deco_model.summary()
output = deco_model(enco_model(input_img))
auto_model = Model(inputs = input_img, outputs = output) #Combining Encoder and Decoder Model as Autoencoder

#RMS, ADAM and SGD optimizer definition
opt = SGD(lr = 0.0001, decay = 1e-8, momentum= 0.9, nesterov = True)
opt1 = Adam(0.00009, decay = 1e-8) #0.0001
rms = RMSprop(lr = 0.00009)

#defining a global graph to be used in Data generator
global model_left
model_left = tf.get_default_graph()
model_left = auto_model

#defining a global graph to be used in Data generator
global model_cant
model_cant = tf.get_default_graph()
import keras.backend as k

#DICE Coefficent definition used as a metric in compilation
def dice_coef(y_pred, y_true):
    inter = k.sum(y_true[:,:,-1]*y_pred[:,:,-1])
    return (2*inter + 1)/(k.sum(y_true[:,:,-1]) + k.sum(y_pred[:,:,-1]) + 1)

#Defining the optimizer, loss function and performance metric for model
auto_model.compile(optimizer = opt1, loss = 'mse', metrics=[dice_coef] )
            
#Model training, Prediction on the test samples, mask generation for further thresholding
my_train_data = MY_data(0,list_train, list_gt, 8, path_train, path_gt, None, None)
hist = auto_model.fit_generator(generator = my_train_data, epochs = 150, verbose = 1)    
#predicting masks on the test_raw images
my_pred_data = MY_data(1,list_test,None, 8,path_test, None, model_cant, model_left)
masks = auto_model.predict_generator(my_pred_data)
masks = masks.reshape(masks.shape[0], image_x, image_y)

#Reading the masks given to be used when calculating IOU
mask_true = []
for i in list_gt:
    img = cv2.imread(path_gt + i,0)
    height, width = img.shape
    img_ = cv2.resize(img, None, fx = image_x/width, fy = image_y/height, interpolation = cv2.INTER_LINEAR)
    img_ =  img_/255
    mask_true.append(img_)
mask_true = np.array(mask_true)         
 
# calculating the thresholds from OTSU and then thresholding masks according to each sample. 
from skimage import filters 
val = np.zeros(masks.shape[0])
for i in range(masks.shape[0]):
    val[i] = filters.threshold_otsu(masks[i])
masks_thres = np.ones((masks.shape[0], image_x, image_y))
for i in range(masks.shape[0]):
    for j in range(masks.shape[1]):
        for l in range(masks.shape[2]):
            if(masks[i][j][l] > val[i]):
                masks_thres[i][j][l] = 0 
   
#IOU score used to evaluate on 15 unseen training samples initially to verify the performance of the model
#iou_1 = np.sum(np.logical_and(mask_true, masks_thres))
#iou_2 =  np.sum(np.logical_or(mask_true, masks_thres))
#iou_scre = iou_1/iou_2
#print(iou_scre)

# Two keys: ['loss', 'dice_coef'], Dice_coeficient graph plot
print(hist.history.keys())
plt.plot(hist.history['dice_coef'])
plt.title('Model performance')
plt.ylabel('Dice Coefficient')
plt.xlabel('Epoch')
plt.show()

#Used to save the masks produced for the predicted test images, after normalizing it to [0, 255] 
#for i in range(len(list_train)):
#    img = cv2.imread(path_train + list_train[i],0)
#    height, width= img.shape
#    mask_img = cv2.resize(masks[i], None, fx = width/image_x, fy = height/image_y, interpolation = cv2.INTER_CUBIC)
#    height_1, width_1 = mask_img.shape
#    norm_img = np.zeros((width_1, height_1))
#    norm_img = cv2.normalize(mask_img, norm_img,0,255, cv2.NORM_MINMAX)
#    cv2.imwrite("D:\\PROJECTS\\mask_raw\\" + list_train[i], norm_img)
#    

path_1 = "D:\\PROJECTS\\mask_raw_3" #Path of training samples
list_t1 = os.listdir(path_1)
mm1 = []
for i in range(len(mm1)):
    img = (cv2.imread(path_1 + '\\'+ list_t1[i],0))
    height, width = img.shape
    imgd = cv2.resize(img, None, fx = image_x/width, fy = image_y/height, interpolation = cv2.INTER_LINEAR)    
    plt.imsave("D:\\PROJECTS\\mask_raw_3\\" + list_t1[i], imgd ,vmin = [0], vmax= [1],cmap = 'binary')
#    cv2.imwrite("D:\\PROJECTS\\mask_raw_3\\" + list_t[i] ,imgd)
    mm1.append(imgd)
mm1 = np.array(mm1)

