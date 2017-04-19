# -*- coding: utf-8 -*-
'''
For setting default values of ANN and DBN.

                                                                    Written by Hyungwon Yang
                                                                                2016. 02. 10
                                                                                   EMCS Labs
'''

import numpy as np
import tensorflow as tf

# Setting the DBN default value.
class setParam(object):

    def __init__(self,inputData,targetData,hiddenUnits):
        self.inputs = inputData
        self.outputs = targetData
        self.hiddenUnits = hiddenUnits

        # check inputs and outputs
        assert self.inputs.shape[0] and self.outputs.shape[0] > 0

        #  initial weight and bias
        self.addrange = 0.1

    def genHiddenBox(self):

        inputDim = self.inputs.shape[1]
        outputDim = self.outputs.shape[1]
        hiddenBox = inputDim
        for iter in self.hiddenUnits:
            hiddenBox = np.append(hiddenBox,iter)

        hiddenBox= np.append(hiddenBox,outputDim)
        return hiddenBox


    def genWeight(self,option='default'):

        hid = self.genHiddenBox()
        weightMatrix = []
        for layer in range(len(hid)-1):
            if option == 'default':
                seed = np.random.randint(100,999)
                self.genRan = np.random.RandomState(seed)
                weight = tf.Variable(self.genRan.rand(hid[layer],hid[layer+1]) * self.addrange * 2 - self.addrange,name="weight")
            elif option == 'random_normal':
                weight = tf.Variable(np.random.normal(0,1,[hid[layer],hid[layer+1]]))
            else:
                print('Option is not properly set. It will be reset automatically as a default.')
                seed = np.random.randint(100, 999)
                self.genRan = np.random.RandomState(seed)
                weight = tf.Variable(self.genRan.rand(hid[layer], hid[layer + 1]) * self.addrange * 2 - self.addrange,
                                     name="weight")
            # Check data type.
            if self.inputs.dtype == 'float64':
                new_weight = tf.cast(weight, tf.float64)

            elif self.inputs.dtype == 'float32':
                new_weight = tf.cast(weight, tf.float32)
            else:
                ValueError('Input data type should be float for input and weight multiplication.')
            weightMatrix.append(new_weight)

        return weightMatrix

    def genBias(self,option='default'):

        hid = self.genHiddenBox()
        biasMatrix = []
        for layer in hid:
            if layer < 2:
                biasInit = 1
            else:
                if option == 'default':
                    biasInit = np.log10((1.0/layer) / (1.0-(1.0/layer)))
                    bias = tf.Variable(np.array([np.tile(biasInit, layer)]), name="bias")
                elif option == 'random_normal':
                    bias = tf.Variable(np.random.normal(0,1,(1,layer)),name='bias')
                else:
                    print('Option is not properly set. It will be reset automatically as a default.')
                    biasInit = np.log10((1.0 / layer) / (1.0 - (1.0 / layer)))
                    bias = tf.Variable(np.array([np.tile(biasInit,layer)]),name="bias")
            # Check data type.
            if self.inputs.dtype == 'float64':
                new_bias = tf.cast(bias, tf.float64)

            elif self.inputs.dtype == 'float32':
                new_bias = tf.cast(bias, tf.float32)
            else:
                ValueError('Input data type should be float for input and weight multiplication.')
            biasMatrix.append(new_bias)

        return biasMatrix


    def setWeight(self,given_weight):
        # Resetting RBM trained weight parameters for next DNN training.
        # However, it can be applied for another purpose.

        weightMatrix = []
        # 1 to N-1 hidden layers.
        for layer in range(len(given_weight)-1):
            weight = tf.Variable(given_weight[layer],name="weight")
            # Check data type.
            if self.inputs.dtype == 'float64':
                new_weight = tf.cast(weight, tf.float64)

            elif self.inputs.dtype == 'float32':
                new_weight = tf.cast(weight, tf.float32)
            else:
                ValueError('Input data type should be float for input and weight multiplication.')
            weightMatrix.append(new_weight)

        # Last Nth hidden layer.
        weightMatrix.append(given_weight[-1])

        return weightMatrix


    def setBias(self,given_bias):
        # Resetting RBM trained bias parameters for next DNN training.
        # However, it can be applied for another purpose.

        biasMatrix = []
        # 1 to N-1 hidden layers.
        for layer in range(len(given_bias)-1):

            bias = tf.Variable(given_bias[layer],name="bias")
            if self.inputs.dtype == 'float64':
                new_bias = tf.cast(bias, tf.float64)

            elif self.inputs.dtype == 'float32':
                new_bias = tf.cast(bias, tf.float32)
            else:
                ValueError('Input data type should be float for input and weight multiplication.')
            biasMatrix.append(new_bias)

        # Last Nth hidden layer.
        biasMatrix.append(given_bias[-1])

        return biasMatrix

    # Generate symbolic matrices.
    # x is training input and y is training output(target).
    def genSymbol(self):
        self.input_x = tf.placeholder(self.inputs.dtype,[None,None])
        self.input_y = tf.placeholder(self.inputs.dtype,[None,None])

        return self.input_x, self.input_y

# Setting the RNN default value.
class RNNParam(object):

    def __init__(self,inputData,targetData,timeStep,hiddenUnits):
        self.inputs = inputData
        self.outputs = targetData
        self.hiddenUnits = hiddenUnits
        self.timeStep = timeStep
        self.inputDim = inputData.shape[2]
        self.outputDim = targetData.shape[2]

        # check inputs and outputs
        assert self.inputs.shape[0] and self.outputs.shape[0] > 0

    def genWeight(self):
        # Check data type.
        if self.inputs.dtype == 'float64':
            tmp_weight = tf.Variable(tf.random_normal([self.hiddenUnits[0], self.outputDim]))
            weight = tf.cast(tmp_weight, tf.float64)
        elif self.inputs.dtype == 'float32':
            tmp_weight = tf.Variable(tf.random_normal([self.hiddenUnits[0], self.outputDim]))
            weight = tf.cast(tmp_weight, tf.float32)
        else:
            ValueError('Input data type should be float for input and weight multiplication.')
        return weight

    def genBias(self):
        # Check data type.
        if self.inputs.dtype == 'float64':
            tmp_bias = tf.Variable(tf.random_normal([self.outputDim]))
            bias = tf.cast(tmp_bias, tf.float64)
        elif self.inputs.dtype == 'float32':
            tmp_bias = tf.Variable(tf.random_normal([self.outputDim]))
            bias = tf.cast(tmp_bias, tf.float32)
        else:
            ValueError('Input data type should be float for input and weight multiplication.')
        return bias


    def genSymbol(self):
        self.input_x = tf.placeholder(self.inputs.dtype, [None, self.timeStep, self.inputDim])
        self.input_y = tf.placeholder(self.inputs.dtype, [None, self.outputDim])

        return self.input_x, self.input_y


def shakeBatch(inputNumber,batchSize,option):

    # Get batchNumber
    if np.remainder(inputNumber,batchSize) is 0:
        batchNumber = (inputNumber/batchSize)
    else:
        batchNumber = (inputNumber/batchSize) + 1

    # Check option
    if option is 'train':
        batchBox = np.random.permutation(inputNumber)
    elif option is 'test':
        batchBox = range(inputNumber)
    else:
        raise Exception('Option variable of shakeBatch function is inapplicable. It should be train or test.')

    # start = -1
    final = -1
    batchIndex = []
    for run in range(batchNumber):

        start = final + 1
        final = final + batchSize
        if final > inputNumber:
            final -= final - inputNumber
        batchIndex.append(batchBox[start:final+1])

    return batchIndex[0:-1]




