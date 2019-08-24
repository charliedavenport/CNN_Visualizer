# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 12:12:55 2019

@author: cdave
"""

from base.base_trainer import BaseTrainer
from models.LeNet_MNIST import LeNet_MNIST
from data_loaders.MNIST_data_loader import MNIST_data_loader
from datetime import datetime
import keras.callbacks
import os

class LeNet_MNIST_trainer(BaseTrainer):
    
    def __init__(self, model=None, data=None, save_csv=False, save_chkpt=False):
        if model == None: 
            model = LeNet_MNIST()
        if data == None: 
            data = MNIST_data_loader().get_train_data()
        super().__init__(model, data)
        self.callbacks = []
        self.loss = []
        self.acc = []
        self.val_loss = []
        self.val_acc = []
        self.init_callbacks(save_csv, save_chkpt)
        
    def init_callbacks(self, save_csv, save_chkpt):
        if save_csv or save_chkpt:
            now = datetime.utcnow().strftime('%Y%m%d-%H%M%S')
            exp_name = "{0}-{1}".format(self.model.name, now)
            logdir = os.path.join('experiments', exp_name)
            os.mkdir(logdir)
            print('Saving experiment data in {}'.format(logdir))
        else:
            print('Not saving training data for this run. Create a new instance '
                  + 'of LeNet_MNIST_trainer with save_csv=True or save_chkpt=True '
                  + 'to save training data to disk.')
        if save_csv:
            self.callbacks.append(keras.callbacks.CSVLogger(os.path.join(logdir, "{0}.csv".format(self.model.name)),
                                                            separator=',', append=True))
        if save_chkpt:
            weights_fname = self.model.name + "-weights.{epoch:02d}-{val_loss:.4f}.hdf5"
            self.callbacks.append(keras.callbacks.ModelCheckpoint(os.path.join(logdir, weights_fname),
                                                                  save_best_only=True))
        print(self.callbacks)
        
        
    def train(self, epochs, batch_size=64, val_split = 0.2, lr_sched=None):
        if lr_sched is not None:
            self.callbacks.append(keras.callbacks.LearningRateScheduler(lr_sched))
        hist = self.model.model.fit(self.data[0], self.data[1], epochs=epochs,
                                    callbacks=self.callbacks, batch_size=batch_size,
                                    validation_split = val_split)
        self.loss.extend(hist.history['loss'])
        self.acc.extend(hist.history['acc'])
        self.val_loss.extend(hist.history['val_loss'])
        self.val_acc.extend(hist.history['val_acc'])
        
if __name__=='__main__':
    trainer = LeNet_MNIST_trainer()
    trainer.train(epochs=1)
                              