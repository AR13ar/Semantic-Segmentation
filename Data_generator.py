# -*- coding: utf-8 -*-
"""
Created on Sun Jul 28 16:57:19 2019

@author: Aditya Raj
"""
import numpy as np
import cv2
import tensorflow as tf
from keras.utils import Sequence
image_x = 320
image_y = 320
#Data Generator
class MY_data(Sequence):
    # Image_filenames_a are the images from training samples folder, Image_filenames_b are the corresponding Ground truth
    def __init__(self, aa,image_filenames_a, image_filenames_b, batch_size, path_train, path_gt, model_cant, model_left):       
        self.image_filenames_a, self.image_filenames_b = image_filenames_a, image_filenames_b
        self.batch_size = batch_size
        self.path_train, self.path_gt = path_train, path_gt
        self.aa = aa
        self.model_cant, self.model_left = model_cant, model_left
    def __len__(self): 
            return int(np.ceil(np.array(len(self.image_filenames_a)/(self.batch_size))))              
    def __getitem__(self, idx):
        # indx = 0, is used for training the Autoencoder model where the function returns the TRAINING IMAGE and its Mask
        if(self.aa == 0):    
            #defining the batch size number of images to be used for training and weight updates at one time
            batch_x = self.image_filenames_a[int(idx)* self.batch_size:(int(idx)+1)* self.batch_size]
            images_a = []
            batch_y = self.image_filenames_b[int(idx)* self.batch_size:(int(idx)+1)* self.batch_size]
            images_b = []
            #reading the training samples using cv2, resizing them to 320 x 320
            for file_name in batch_x:            
                img = cv2.imread(self.path_train + file_name ,0)
                height, width = img.shape
                img_ = cv2.resize(img, None, fx = image_x/width, fy = image_y/height, interpolation = cv2.INTER_LINEAR)
                img_ =  img_/255                
                images_a.append(img_)
            images_a = np.array(images_a)
            # The noise can be added to the training samples to make the model robust, initially a noise factor of 0.2 was added
            noise_factor = 0
            #Preprocessing the train samples by normalizing around mean
            images_a = (images_a.mean() - images_a)/images_a.std()
            images_a_noise = images_a + noise_factor*np.random.normal(loc = 0.0, scale = 1.0)
            images_a = images_a.reshape((images_a.shape[0]), image_x, image_y, 1)
            images_a_noise = images_a_noise.reshape((images_a_noise.shape[0]), image_x, image_y,1)            
            
            #reading the Masks corresponding to the training samples using cv2, resizing them to 320 x 320
            for file_name in batch_y:
                img = cv2.imread(self.path_gt + file_name ,0)
                height, width = img.shape
                img_ = cv2.resize(img, None, fx = image_x/width, fy = image_y/height, interpolation = cv2.INTER_LINEAR)
                img_ =  img_/255                
                images_b.append(img_)                
            images_b = np.array(images_b)                
            #Preprocessing the mask samples by normalizing around mean
            images_b = (images_b.mean() - images_b)/images_b.std()
            images_b = images_b.reshape(images_b.shape[0], image_x, image_y,1 )            
            return [images_a_noise,images_b]

        # indx = 1, returns the Masks generated after the network is trained in a batch size of 8        
        if(self.aa == 1):
            batch_x = self.image_filenames_a[int(idx)* self.batch_size:(int(idx)+1)* self.batch_size]
            images_a = []
            images_pred_a = []  
            #reading the test samples using cv2, resizing them to 320 x 320
            for file_name in batch_x:            
                img = cv2.imread(self.path_train + file_name,0)
                height, width = img.shape
                img_ = cv2.resize(img, None, fx = image_x/width, fy = image_y/height, interpolation = cv2.INTER_LINEAR)
                img_ =  img_/255   
                images_a.append(img_)
            images_a = np.array(images_a)
            #Preprocessing the test samples by normalizing around mean
            images_a = (images_a.mean() - images_a)/images_a.std()
            #Reshaping the test images to be further used to predict masks
            images_a = images_a.reshape(images_a.shape[0], image_x, image_y,1)
            #this function outputs the Masks predicted by the model
            with self.model_cant.as_default():
                model_cant = self.model_left
                images_pred_a.append(model_cant.predict(images_a))
                self.model_cont = tf.get_default_graph()
            images_pred_a = np.array(images_pred_a)
            images_pred_a = images_pred_a.reshape(images_a.shape[0], image_x, image_y,1)          
            return images_pred_a