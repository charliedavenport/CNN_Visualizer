# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 14:02:42 2019

@author: cdave
"""

from models.LeNet_MNIST import LeNet_MNIST
from data_loaders.MNIST_data_loader import MNIST_data_loader

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

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

        w = 1/n_layer # unit width allotted for each layer
        self.im_width = 0.8*w
        self.arr_width = 0.2*w
        
        #print("{} {}".format(self.im_width, self.arr_width)) #debug
        
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
            new_ax.axis('off')
            
            # ARROW
            left = (i+2) * self.im_width + (i+1) * self.arr_width
            self.add_arrow(left)
            
        # DENSE LAYER ACTIVATIONS
        for i in range(self.model.dense_layers-1):
            i += self.model.conv_layers
            left = (i+1)*(self.im_width + self.arr_width)
            new_ax = self.fig.add_axes([left,1.0, self.im_width,1.0])
            self.layer_axes.append(new_ax)
            #i -= self.model.conv_layers
            new_ax.set_title('Dense Layer {}'.format(i+1), fontdict=self.font_dict)
            new_ax.axis('off')
            
            # ARROW
            left = (i+2) * self.im_width + (i+1) * self.arr_width
            self.add_arrow(left)
            
        # OUTPUT LAYER
        # debug - random numbers for output layer
        #pred = np.random.rand(1,10)
        left = (n_layer) *(self.im_width + self.arr_width)
        new_ax = self.fig.add_axes([left,1.0, self.im_width,1.0])
        self.layer_axes.append(new_ax)
        #new_ax.matshow(pred)
        new_ax.set_title('Output Layer', fontdict=self.font_dict)
        new_ax.axis('on')
        new_ax.set_yticks([])
        new_ax.set_ylabel('')
        new_ax.set_xticks(range(10))
        new_ax.xaxis.set_ticks_position('bottom')
        new_ax.spines['bottom'].set_color('black')
        new_ax.tick_params(axis='x', colors='black')
        # debug
        #self.fig.savefig("test.png", dpi=150, bbox_inches='tight', facecolor='white')
        
    def add_arrow(self, left):
        new_ax = self.fig.add_axes([left,1.0, self.arr_width,1.0])
        arrow_tail = (0.0, 0.5)
        arrow_head = (1.0, 0.5)
        arrow = mpatches.FancyArrowPatch(arrow_tail, arrow_head,
                                         mutation_scale=40, facecolor='white')
        new_ax.add_patch(arrow)
        new_ax.axis('off')
        
    def output_text(self, prediction, confidence):
        ax = self.layer_axes[-1]
        fd_b = {'fontsize' : 48, 
                'weight' : 'bold',
                'color' : 'black'}
        if confidence > 50.0: 
            ax.text(4,4, '{0}'.format(prediction), fontdict=fd_b)
        else:
            ax.text(4,4, '[?]', fontdict=self.font_dict)
        ax.text(0,5, 'Confidence: {0:.6f}%'.format(confidence), fontdict=self.font_dict_s)
    
    def plot_layer_outputs(self, input_img):
        for i, ax in enumerate(self.layer_axes):
            if i==0:
                ax.imshow(input_img.reshape(28,28), cmap='Greys')
            else:
                activations = self.model.get_layer_outputs(i-1, input_img)
                ax.imshow(activations)
                if i == len(self.layer_axes) - 1:
                    prediction = np.argmax(activations[0])
                    confidence = activations[0, prediction] * 100
                    self.output_text(prediction, confidence)
                
            
    
if __name__=="__main__":
    #plt.ion()
    model = LeNet_MNIST(from_chkpt=True,
                        chkpt_path='..\experiments\LeNet_MNIST-20190823-210500\LeNet_MNIST-weights.03-0.0895.hdf5')
    
    dl = MNIST_data_loader()
    X_test, y_test = dl.get_test_data()
    #loss, acc = model.model.evaluate(X_test, y_test)
    #print('loss = {}, acc = {}'.format(loss, acc))
    
    newFig = LeNet_MNIST_Figure(model, dl)
    
    rand_ind = np.random.randint(X_test.shape[0])
    test_img = X_test[rand_ind]
    newFig.plot_layer_outputs(test_img)
    #plt.show()
    #newFig.fig.show()
    
    newFig.fig.savefig('test_{}.png'.format(rand_ind), bbox_inches='tight')

        
    
        