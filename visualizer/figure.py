# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 14:02:42 2019

@author: cdave
"""

from models.LeNet_MNIST import LeNet_MNIST
from data_loaders.MNIST_data_loader import MNIST_data_loader

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from keras.models import load_model

class LeNet_MNIST_Figure(object):
    
    
    def __init__(self, model, data_loader):
        self.model = model
        self.data_loader = data_loader
        self.fig = None
        self.layer_axes = []
        
        self.init_figure()
        
    def init_figure(self):
        
        
       
        
        n_layer = self.model.conv_layers + self.model.dense_layers
        
        self.fig = plt.figure(figsize=(5 * n_layer, 5))

        w = 1/n_layer
        self.im_width = 0.8*w
        self.arr_width = 0.2*w
        
        self.font_dict = {'fontsize' : 16,
                     'color' : 'black',
                     'weight' : 'bold'}
        self.font_dict_s = {'fontsize' : 12,
                       'color' : 'black'}
        
        # INPUT IMAGE
        # debug - comment this line out
        test_img = self.data_loader.get_test_data()[0][0].reshape((28,28))
        
        input_ax = self.fig.add_axes([0.0,1.0, self.im_width,1.0])
        input_ax.imshow(test_img, cmap='Greys')
        #input_ax.axis('off')
        input_ax.set_title('Input', fontdict=self.font_dict)
        self.layer_axes.append(input_ax)
        
        # ARROW
        left = self.im_width
        self.add_arrow(left)
        
        # CONV LAYER ACTIVATIONS
        for i in range(self.model.conv_layers):
            left = (i+1)*(self.im_width + self.arr_width)
            new_ax = self.fig.add_axes([left,1.0, self.im_width,1.0])
            self.layer_axes.append(new_ax)
            new_ax.set_title('Convolutional Layer {}'.format(i+1), fontdict=self.font_dict)
            #new_ax.axis('off')
            
            # ARROW
            left = (i+2) * self.im_width + (i+1) * self.arr_width
            self.add_arrow(left)
            
        # DENSE LAYER ACTIVATIONS
        for i in range(self.model.dense_layers):
            i += self.model.conv_layers
            left = (i+1)*(self.im_width + self.arr_width)
            new_ax = self.fig.add_axes([left,1.0, self.im_width,1.0])
            self.layer_axes.append(new_ax)
            #i -= self.model.conv_layers
            new_ax.set_title('Dense Layer {}'.format(i+1), fontdict=self.font_dict)
            #new_ax.axis('off')
            # ARROW
            left = (i+2) * self.im_width + (i+1) * self.arr_width
            self.add_arrow(left)
            
        self.fig.savefig("test.png", dpi=150, bbox_inches='tight', facecolor='white')
        
    def add_arrow(self, left):
        new_ax = self.fig.add_axes([left,1.0, self.arr_width,1.0])
        arrow_tail = (0.0, 0.5)
        arrow_head = (1.0, 0.5)
        arrow = mpatches.FancyArrowPatch(arrow_tail, arrow_head,
                                         mutation_scale=40, facecolor='white')
        new_ax.add_patch(arrow)
        new_ax.axis('off')
        
        
    def plot_layer_outputs(self, input_img):
        pass
    
if __name__=="__main__":
    model = LeNet_MNIST()
    model.load_weights('..\experiments\LeNet_MNIST-20190823-175823\LeNet_MNIST-weights.03-0.1166.hdf5')
    dl = MNIST_data_loader()
    newFig = LeNet_MNIST_Figure(model, dl)

        
    
        