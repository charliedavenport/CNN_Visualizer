# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 12:15:27 2019

@author: cdave
"""

from base.base_data_loader import BaseDataLoader
from keras.datasets import mnist
import numpy as np
import keras.utils

class MNIST_data_loader(BaseDataLoader):
    
    def __init__(self):
        super().__init__()
        (self.X_train, self.y_train), (self.X_test, self.y_test) = mnist.load_data()
        self.X_train = np.expand_dims(self.X_train.astype('float32') / 255, axis=-1)
        self.X_test = np.expand_dims(self.X_test.astype('float32') / 255, axis=-1)
        self.y_train = keras.utils.to_categorical(self.y_train)
        self.y_test = keras.utils.to_categorical(self.y_test)
        
        
    def get_train_data(self):
        return (self.X_train, self.y_train)
    
    def get_test_data(self):
        return (self.X_test, self.y_test)