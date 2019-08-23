# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 10:39:42 2019

@author: cdave

This script creates a simple leNet model and trains it on MNIST.
No visualization is done here. To do that, run the main module in /visualizer
"""

from models.LeNet_MNIST import LeNet_MNIST
from data_loaders.MNIST_data_loader import MNIST_data_loader
from trainers.LeNet_MNIST_trainer import LeNet_MNIST_trainer


def main():
    print("Creating leNet model...")
    lenet_model = LeNet_MNIST()
    
    print("Training leNet model on MNIST...")
    mnistDL = MNIST_data_loader()
    trainer = LeNet_MNIST_trainer(lenet_model, 
                                  mnistDL.get_train_data())
    trainer.train(epochs=10)


if __name__=="__main__":
    main()