# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 11:14:14 2019

@author: cdave

LeNet model: http://yann.lecun.com/exdb/lenet/
"""

from base.base_model import BaseModel
from keras.layers import (Dense, Conv2D, MaxPooling2D, 
                          Dropout, Flatten)
from keras.models import Sequential
import keras.utils

class LeNet_MNIST(BaseModel):
    
    name = "LeNet_MNIST"
    
    # model hyperparams
    INPUT_SHAPE = (28,28,1)
    KERNEL_SIZE = (5,5)
    KERNEL_STRIDES = (1,1)
    KERNEL_INIT = 'truncated_normal'
    POOL_SIZE = (2,2)
    POOL_STRIDES = (2,2)
    ACTIV = 'relu'
    LEARN_RATE = 0.01
    
    def __init__(self):
        super().__init__()
        self.build_model()
        
    def build_model(self):
        self.model = Sequential()
        ## C1 ##
        self.model.add(Conv2D(6, self.KERNEL_SIZE, strides=self.KERNEL_STRIDES,
                              kernel_initializer=self.KERNEL_INIT, 
                              activation=self.ACTIV,
                              input_shape=(self.INPUT_SHAPE),
                              padding='same'))
        ## S2 ##
        self.model.add(MaxPooling2D(pool_size=self.POOL_SIZE,
                                    strides=self.POOL_STRIDES))
        ## C3 ##
        self.model.add(Conv2D(16, self.KERNEL_SIZE, strides=self.KERNEL_STRIDES,
                              kernel_initializer=self.KERNEL_INIT,
                              activation=self.ACTIV))
        ## S4 ##
        self.model.add(MaxPooling2D(pool_size=self.POOL_SIZE,
                                    strides=self.POOL_STRIDES))
        ## C5 ##
        self.model.add(Conv2D(120, self.KERNEL_SIZE, strides=self.KERNEL_STRIDES,
                              kernel_initializer=self.KERNEL_INIT,
                              activation=self.ACTIV))
        self.model.add(Flatten())
        ## F6 ##
        self.model.add(Dense(84, activation=self.ACTIV, kernel_initializer=self.KERNEL_INIT))
        self.model.add(Dropout(0.5))
        ## OUT ##
        self.model.add(Dense(10, activation='softmax'))
        
        self.model.compile(loss=keras.losses.categorical_crossentropy,
                           optimizer=keras.optimizers.Adam(lr=self.LEARN_RATE),
                           metrics=['accuracy'])
        print(self.model.summary())