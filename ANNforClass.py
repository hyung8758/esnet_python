# -*- coding: utf-8 -*-
'''
Artificial Neural Network for Classification and Curvefitting problems.

This script is written for activating a simple artificial neural network. (ANN)
Please refer to the step as follows.

1. Import a sample example set from Matlab.
    * You need to understand the structure and the characteristics of the inputs and outputs
2.


                                                                    Written by Hyungwon Yang
                                                                                2016. 02. 07
                                                                                    EMCS Lab
'''

import numpy as np
import theano
import theano.tensor as T
import loadfile

from SetValues import SetValues
from ANN_model import ANN_model



def ann_train():

    print 'Loading the data...'

    # Import cancerData
    train_in, train_out, test_in, test_out = loadfile.readcancer()

    # Import MNIST (optional)
    #train_in, train_out, test_in, test_out = loadfile.readmnist()

    print 'Assigning the data variables...'

    # Generate initial values.
    values = SetValues(inputs=train_in,
                       outputs=train_out,
                       learningRate=0.001,
                       momentum=0.9,
                       epochNum=1000,
                       hiddenUnits=50,
                       W=None,
                       b=None)
    ihMatrix,hoMatrix = values.genWeight()
    hBiasMatrix, oBiasMatrix = values.genBias()
    inX, outY = values.genMatrices()

    batchSize = 10

    print 'Constructing ANN_model...'

    models = ANN_model(inX=inX,
                       outY=outY,
                       ihMatrix=ihMatrix,
                       hoMatrix=hoMatrix,
                       hBiasMatrix=hBiasMatrix,
                       oBiasMatrix=oBiasMatrix,
                       learningRate=0.001)

    outputStorage, outputActivation = models.fnn(methods='classification')
    cost, updates = models.sgd(outputStorage)

    train = theano.function(inputs=[inX,outY],outputs=cost,updates=updates,allow_input_downcast=True)
    predict = theano.function(inputs=[inX],outputs=outputActivation,allow_input_downcast=True)

    print 'Training the data...'

    total_epoches = range(values.epoches())
    error_history = []
    for epoch in total_epoches:

        # Shuffling inputs and outputs


        #total_inputs = range(train_in.shape[0])
        for batch_in,batch_out in zip(range(0,train_in.shape[0],batchSize),range(batchSize,train_in.shape[0],batchSize)):
        #for iter in total_inputs:
            error = train(train_in[batch_in:batch_out],train_out[batch_in:batch_out])
            error_history.append(error)

        # print error rate and prediction correctness
        error_sum = np.mean(error_history)
        print 'Epoch: {}, error: {}.'.format(epoch+1,error_sum)
        accuracy = np.mean(np.argmax(test_out,axis=1) == predict(test_in))
        print '\tAccuracy: {} percent'.format(accuracy)


    print '\nprocess finished.\n'


if __name__ == '__main__':
    ann_train()