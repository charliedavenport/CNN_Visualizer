# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 11:14:14 2019

@author: cdave

LeNet model: http://yann.lecun.com/exdb/lenet/
"""

from base.base_model import BaseModel
from data_loaders.MNIST_data_loader import MNIST_data_loader

from keras.layers import (Dense, Conv2D, MaxPooling2D, 
                          Dropout, Flatten)
from keras.models import Sequential, Model
import keras.utils
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

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
    
    conv_layers = 3
    dense_layers = 2
    
    def __init__(self, from_chkpt=False, chkpt_path=None, from_config=False, save_config=False, 
                 config_dir='configs', config_name=None):
        super().__init__()
        if from_chkpt and chkpt_path is not None:
            self.load_checkpoint(chkpt_path)
        elif from_config and config_name is not None:
            config_path = os.path.join(config_dir, config_name)
            self.build_from_config(config_path)    
        else:
            self.build_model(save_config, config_dir)
            
        self.init_layer_models()
        
        
    def build_model(self, save_config, config_dir):
        
        print('\nBuilding new model...')
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
        print('Model Summary:')
        print(self.model.summary())
        if save_config:
            now = datetime.utcnow().strftime('%Y%m%d-%H%M%S')
            config_path = os.path.join(config_dir, '{0}-{1}.json'.format(self.name, now))
            print('Saving configuration in {}'.format(config_path))
            self.export_config(config_path)
        else:
            print('Not saving the config for this file. To save model config, pass in'
                  + ' save_config=True')
            
    def init_layer_models(self):
        self.layer_models = []
        for layer in self.model.layers:
            name = layer.get_config()['name']
            # only get the conv2d and dense layers
            if ('conv2d' in name or 'dense' in name):
                self.layer_models.append(Model(inputs=self.model.input,
                                               outputs=layer.output))
                
    def get_n_layers(self):
        return len(self.layer_models)
        
    
    def get_layer_outputs(self, layer_ind, input_img):
        if (layer_ind > len(self.layer_models)):
            print('LeNet_MNIST.get_layer_outputs: invalid layer_ind')
            return
        if (len(input_img.shape) < 4):
            input_img = np.expand_dims(input_img, axis=0)
        layer_model = self.layer_models[layer_ind]
        pred = layer_model.predict(input_img)
        if layer_ind < self.conv_layers:
            # conv layer
            print('output shape = {}'.format(pred.shape))
            n_filter = pred.shape[-1]  
            n_row = int(np.ceil(np.sqrt(n_filter)))
            n_col = n_filter // n_row
            #print(n_row, n_col)
            im_size = pred.shape[1:3]
            #print(im_size)
            activations_grid = np.empty(shape=(im_size[0] * n_row, 
                                               im_size[1] * n_col))
            #print(activations_grid.shape)
            filter_index = 0
            for i in range(n_row):
                for j in range(n_col):
                    x = im_size[0] # store the image size for indexing
                    activations_grid[i*x: (i+1)*x, 
                                     j*x: (j+1)*x] = pred[0,:,:, filter_index]
                    filter_index += 1

            return activations_grid
        else:
            # dense layer (including output layer)
            if pred.size > 10:
                pred = pred.reshape(12, -1)
            return pred
            
            
        



if __name__=="__main__":
    # Test the LeNet_MNIST class:
    # from config
    #config_dir = '..\configs'
    #config_name = 'LeNet_MNIST-20190824-150631.json'
    #lenet = LeNet_MNIST(from_config=True, save_config=False, 
    #            config_dir=config_dir, config_name=config_name)
    
    # load model checkpoint
    lenet = LeNet_MNIST(from_chkpt=True, 
                        chkpt_path='..\experiments\LeNet_MNIST-20190823-210500\LeNet_MNIST-weights.03-0.0895.hdf5')
    
#    test_img = MNIST_data_loader().get_test_data()[0][1]
#    test_img = np.expand_dims(test_img, axis=0)
#    conv_1_out = lenet.layer_models[0].predict(test_img)
#    print(conv_1_out[0,:,:,0])
#    for i in range(6):
#        plt.imshow(conv_1_out[0,:,:,i])
#        plt.show()
    print('LAYER 1 FILTERS (6)')
    for i in range(6):
        # shape (5,5,1,6)
        print('Convolutional filter {}'.format(i))
        plt.imshow(lenet.model.layers[0].get_weights()[0][:,:,0,i])
        print('bias = {}'.format(lenet.model.layers[0].get_weights()[1][i]))
        plt.show()
    
    
    
    # load weights
    #lenet.load_weights('..\experiments\LeNet_MNIST-20190823-210500\LeNet_MNIST-weights.03-0.0895.hdf5')
    
    # test layer outputs
    test_img = MNIST_data_loader().get_test_data()[0][1]
    for ind in range(lenet.get_n_layers()):
        result = lenet.get_layer_outputs(ind, test_img)
        print(lenet.layer_models[ind].layers[-1].get_config()['name'])
        plt.imshow(result)
        plt.show()
    
    # from scratch
    #LeNet_MNIST(False, False, config_dir, None)
    