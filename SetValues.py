# -*- coding: utf-8 -*-
'''
Artificial Neural Network for Classification and Curvefitting problems.

This script is written for activating a simple artificial neural network. (ANN)
Please refer to the step as follows.

1. Import a sample example set from Matlab.
    * You need to understand the structure and the characteristics of the inputs and outputs
2.


                                                                    Written by Hyungwon Yang
                                                                                2016. 02. 10
                                                                                    EMCS Lab
'''
import numpy as np
import theano
import theano.tensor as T


class SetValues(object):

    def __init__(self,inputs,outputs,learningRate=0.001,
                 momentum=0.9,epochNum=25,hiddenUnits=50,W=None,b=None):
        self.inputs = inputs
        self.outputs = outputs
        self.lr = learningRate
        self.momentum = momentum
        self.epochNum = epochNum
        self.hiddenUnits = hiddenUnits
        self.W = W
        self.b = b

        # check inputs and outputs
        assert len(self.inputs) and len(self.outputs) > 0

        #  initial weight and bias
        self.genRan_w_ih = np.random.RandomState(3493)
        self.genRan_w_ho = np.random.RandomState(9244)
        self.addrange = 0.1

    def epoches(self):
        return self.epochNum

    def floatX(self,value):
        changed_value = np.asarray(value, dtype=theano.config.floatX)
        return changed_value

    # Generate weights.
    def genWeight(self):

        if self.W is None:
            sampleNum,unitNum = self.inputs.shape
            outputSize = self.outputs.shape[1]
            ihMatrix = theano.shared(self.floatX(self.genRan_w_ih.rand(unitNum,self.hiddenUnits) * self.addrange * 2 - self.addrange))
            hoMatrix = theano.shared(self.floatX(self.genRan_w_ho.rand(self.hiddenUnits,outputSize) * self.addrange * 2 - self.addrange))
            return ihMatrix, hoMatrix

    # Generate biases.
    def genBias(self):
        if self.b is None:
            outputSize = self.outputs.shape[1]
            hBiasMatrix = theano.shared(self.floatX(np.ones(self.hiddenUnits)))
            oBiasMatrix = theano.shared(self.floatX(np.ones(outputSize)))
            return hBiasMatrix, oBiasMatrix

    # Generate symbolic matrices.
    def genMatrices(self):
        self.inX = T.fmatrix('inputs')
        self.outY = T.fmatrix('outputs')
        return self.inX, self.outY