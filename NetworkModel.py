# -*- coding: utf-8 -*-
'''
NetworkModel for constructing ANN and DBN models
Feedforward networks, gradient descent.


                                                                    Written by Hyungwon Yang
                                                                                2016. 02. 07
                                                                                    EMCS Lab
'''
import numpy as np
import theano.tensor as T

# This is for ANN
class ANN_model(object):

    def __init__(self,inX,outY,ihMatrix,hoMatrix,hBiasMatrix,oBiasMatrix,learningRate=0.001):
        self.inputs = inX
        self.targets = outY
        self.ihMatrix = ihMatrix
        self.hoMatrix = hoMatrix
        self.hBiasMatrix = hBiasMatrix
        self.oBiasMatrix = oBiasMatrix
        self.lr = learningRate

    # Feedforward Neural Network.
    def fnn(self,methods):
        '''
        You have to choose the methods.
        methods : classification and curvefitting
        '''
        hid = T.nnet.sigmoid(T.dot(self.inputs,self.ihMatrix) + self.hBiasMatrix)
        if methods is 'classification':
            outputStorage = T.nnet.softmax(T.dot(hid,self.hoMatrix) + self.oBiasMatrix)
            outputActivation = T.argmax(outputStorage,axis=1)

        elif methods is 'curvefitting':
            outputStorage = T.dot(hid,self.hoMatrix) + self.oBiasMatrix
            outputActivation = outputStorage

        # estimated inputs
        return outputStorage, outputActivation

    # Stochastic Gradient Descent. (classification)
    def sgd(self,outputStorage):
        cost = T.mean(T.nnet.categorical_crossentropy(outputStorage,self.targets))
        params = [self.ihMatrix,self.hoMatrix,self.hBiasMatrix,self.oBiasMatrix]
        gradient = T.grad(cost=cost, wrt=params)
        updates = []
        for p, g in zip(params, gradient):
            updates.append([p, p - g * self.lr])
        return cost, updates

    # Stochastic Gradient Descent. (curvefitting)
    def sgd2(self,outputStorage):

        cost = T.mean(T.pow(outputStorage-self.targets,2))/2
        params = [self.ihMatrix,self.hoMatrix,self.hBiasMatrix,self.oBiasMatrix]
        gradient = T.grad(cost=cost,wrt=params)
        updates = []
        for p, g in zip(params, gradient):
            updates.append([p, p - g * self.lr])
        return cost, updates


# This is for DBN
class DBN_model(object):

    def __init__(self,inX,outY,weightMatrix,biasMatrix,learningRate):
        self.inputs = inX
        self.targets = outY
        self.weightMatrix = weightMatrix
        self.biasMatrix = biasMatrix
        self.lr = learningRate if learningRate is not None else 0.01

    def fnn(self,methods):
        '''
        This feedforward network is not intended for batch_learning. Instead, it is on-lines based learning networks
        that take more learning time but update weights and biases frequently.
        '''

        # iteration
        hiddenNum = len(self.weightMatrix)
        outputStorage = []
        outputActivation = []

        # First up
        outputStorage.append(T.dot(self.inputs,self.weightMatrix[0]) + self.biasMatrix[1])
        outputActivation.append(T.nnet.sigmoid(outputStorage[0]))

        # Hidden up
        for iter in range(1,hiddenNum-1):

            # Hiddenlayer activation.
            outputStorage.append(T.dot(outputActivation[iter-1],self.weightMatrix[iter]) + self.biasMatrix[iter+1])
            outputActivation.append(T.nnet.sigmoid(outputStorage[iter]))

        # Last up to output: classification.
        if methods is 'classification':


            outputStorage.append(T.nnet.softmax(T.dot(outputActivation[-1],self.weightMatrix[-1]) + self.biasMatrix[-1]))
            outputActivation.append(T.argmax(outputStorage[-1],axis=1))


        # Last up to output: curvefitting.
        elif methods is 'curvefitting':

            outputStorage.append(T.dot(outputActivation[-1],self.weightMatrix[-1]) + self.biasMatrix[-1])
            outputActivation.append(outputStorage[-1])

        return outputStorage, outputActivation

    # Stochastic Gradient Descent. (classification)
    #def sgd(self,outputStorage):
    #    cost = T.mean(T.nnet.categorical_crossentropy(outputStorage,self.targets))
    #    params = [self.ihMatrix,self.hoMatrix,self.hBiasMatrix,self.oBiasMatrix]
    #    gradient = T.grad(cost=cost, wrt=params)
    #    updates = []
    #    for p, g in zip(params, gradient):
    #        updates.append([p, p - g * self.lr])
    #    return cost, updates

    # Stochastic Gradient Descent. (curvefitting)

    def sgd2(self,outputStorage):

        cost = T.mean(T.pow(outputStorage[-1]-self.targets,2))/2
        params = []

        # Set the parameters
        for w in self.weightMatrix[::-1]: params.append(w)
        for b in self.biasMatrix[:0:-1]: params.append(b)

        gradient = T.grad(cost=cost,wrt=params)
        updates = []
        for p, q in zip(params, gradient):
            updates.append([p, p - q * self.lr])

        return cost, updates








