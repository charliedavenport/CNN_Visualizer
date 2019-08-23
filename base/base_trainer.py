# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 10:37:22 2019


https://github.com/Ahmkel/Keras-Project-Template
"""

class BaseTrain(object):
    def __init__(self, model, data):
        self.model = model
        self.data = data
        #self.config = config

    def train(self):
        raise NotImplementedError