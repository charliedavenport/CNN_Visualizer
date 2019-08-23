# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 10:36:18 2019


https://github.com/Ahmkel/Keras-Project-Template
"""

class BaseDataLoader(object):
    def __init__(self):
        #self.config = config

    def get_train_data(self):
        raise NotImplementedError

    def get_test_data(self):
        raise NotImplementedError