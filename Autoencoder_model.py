# -*- coding: utf-8 -*-
"""
@author: Aditya Raj
"""

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, GaussianNoise, Flatten, Dropout, Lambda, Add
from keras.models import Model, Sequential
from keras.layers.advanced_activations import LeakyReLU

image_x = 320
image_y = 320
input_shape = (image_x, image_y,1) #Input For Encoder
input_img = Input(shape= input_shape)
size_ls = (40,40,256)   #Latent Space
size_input = Input(size_ls)

#Encoder Model
def enco(Input_shape):
    autoencoder_left_enco1_1 = (Conv2D(64, (3,3),input_shape = input_shape,name = 'enco_left1_1', padding = 'same', activation = 'relu',  use_bias = True, 
                                kernel_initializer='glorot_uniform'))(Input_shape)
    autoencoder_left_enco = (MaxPooling2D((2,2), name = 'enco_left1_2'))(autoencoder_left_enco1_1)
    autoencoder_left_enco2_1 = (Conv2D(128, (3,3), name = 'enco_left2_1',padding = 'same', activation = 'relu',  use_bias = True, 
                                kernel_initializer='glorot_uniform'))(autoencoder_left_enco)
    autoencoder_left_enco = (MaxPooling2D((2,2),padding = 'same', name = 'enco_left3'))(autoencoder_left_enco2_1)
    autoencoder_left_enco3_1 = (Conv2D(128, (3,3), name = 'enco_left3_1',padding = 'same', activation = 'relu',  use_bias = True, 
                                kernel_initializer='glorot_uniform' ))(autoencoder_left_enco)
    autoencoder_left_enco = (MaxPooling2D((2,2),padding = 'same', name = 'enco_left4'))(autoencoder_left_enco3_1)
    autoencoder_left_enco = (Conv2D(256, (3,3), name = 'enco_left5', activation = 'relu', padding = 'same',  use_bias = True, 
                                kernel_initializer='glorot_uniform'))(autoencoder_left_enco)
    return autoencoder_left_enco

#Decoder Model    
def deco(Input_sha):
    autoencoder_left_deco = Sequential()
    autoencoder_left_deco.add(Conv2D(256, (3,3),input_shape = size_ls, name = 'deco_left1',activation = 'relu' , padding = 'same',  use_bias = True, 
                                kernel_initializer='glorot_uniform'))
    autoencoder_left_deco.add( UpSampling2D((2,2), name = 'deco_left2_'))
    autoencoder_left_deco.add( Conv2D(128, (3,3), name = 'deco_left3_',padding = 'same', activation = LeakyReLU(alpha = 0.3),   use_bias = True, 
                                 kernel_initializer='glorot_uniform')) 
    autoencoder_left_deco.add( UpSampling2D((2,2), name = 'deco_left2'))
    autoencoder_left_deco.add( Conv2D(128, (3,3), name = 'deco_left3_1',padding = 'same', activation = LeakyReLU(alpha = 0.3),   use_bias = True, 
                                 kernel_initializer='glorot_uniform'))
    autoencoder_left_deco.add(UpSampling2D((2,2),  name = 'deco_left4_1'))
    autoencoder_left_deco.add(Conv2D(64, (3,3), name = 'deco_left5',padding = 'same', activation = LeakyReLU(alpha = 0.3),  use_bias = True, 
                                kernel_initializer='glorot_uniform'))
    autoencoder_left_deco.add(Conv2D(1, (3,3), name = 'deco_left6', activation = 'sigmoid', padding = 'same', 
                                     use_bias = True, kernel_initializer='glorot_uniform'))
    autoencoder_left_deco.summary()
    return autoencoder_left_deco
