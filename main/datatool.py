# -*- coding: utf-8 -*-
'''
Manipulating data for them to be fit into training process.
This code was refered from

                                                                    Written by Hyungwon Yang
                                                                                2016. 02. 10
                                                                                   EMCS Labs
'''

import numpy as np

def normalize(data,type,axis=0):
    if type == 'sigmoid':
        tmp_data = 1 / (1 + np.exp(-data))
        normalized_data = [tmp_data]
    elif type == 'tanh':
        tmp_data = np.tanh(data)
        normalized_data = [tmp_data]
    elif type == 'zscore':
        mu = np.mean(data, axis, keepdims=True)
        std = np.std(data, axis, keepdims=True)
        out = (data - mu) / std
        normalized_data = [out, mu, std]
    elif type == 'minmax':
        min_val = np.amin(data, axis=axis)
        max_val = np.amax(data, axis=axis)
        out = (data - min_val) * 2 / max_val - 1
        normalized_data = [out, min_val, max_val]

    return normalized_data

def momentumSigmoid(data,momentum):
    return 1 / (1 + np.exp(-momentum * data))