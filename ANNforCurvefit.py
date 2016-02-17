# -*- coding: utf-8 -*-
'''
Artificial Neural Network for Classification and Curvefitting problems.

This script is written for activating a simple artificial neural network. (ANN)
Please refer to the step as follows.

IT needs to be built!!!!!!
for curvefitting


                                                                    Written by Hyungwon Yang
                                                                                2016. 02. 10
                                                                                    EMCS Lab
'''
from random import choice

import numpy as np
import theano
import theano.tensor as T
import loadfile

from SetValues import SetValues
from ANN_model import ANN_model



def anncurvefit_train():

    print 'Loading the data...'

    # Import cancerData
    train_in, train_out, test_in, test_out = loadfile.readartmfcc()

    # Import bodyfat (optional)
    #train_in, train_out, test_in, test_out = loadfile.readbody()

    # Import building (optional)
    #train_in, train_out, test_in, test_out = loadfile.readbuilding()

    print 'Assigning the data variables...'

    # Generate initial values.
    values = SetValues(inputs=train_in,
                       outputs=train_out,
                       learningRate=0.001,
                       momentum=0.9,
                       epochNum=100,
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

    outputStorage, outputActivation = models.fnn(methods='curvefitting')
    cost, updates = models.sgd2(outputStorage)

    train = theano.function(inputs=[inX,outY],outputs=cost,updates=updates,allow_input_downcast=True)
    predict = theano.function(inputs=[inX],outputs=outputActivation,allow_input_downcast=True)

    print 'Training the data...'

    total_epoches = range(values.epoches())
    input_epoches = len(train_in)/batchSize
    for epoch in total_epoches: # 총 몇번인가.. 그랜드 에포크 한 20번정도? 여기는 모든 인풋데이터가 한번 도는걸 의미함

        error_history = []
        # Shuffling inputs and outputs # 인풋 데이터 돌리기전에 한번 순서들을 섞어줘야함
        rand_num = np.random.permutation(len(train_in))
        train_in = train_in[rand_num]
        train_out = train_out[rand_num]

        #total_inputs = range(train_in.shape[0])
        for batch_in,batch_out in zip(range(0,train_in.shape[0],batchSize),range(batchSize,train_in.shape[0],batchSize)):
        #for iter in total_inputs:
            error = train(train_in[batch_in:batch_out],train_out[batch_in:batch_out])
            error_history.append(error)

        error_sum = np.mean(error_history)

        # print error rate and prediction correctness
        #error_sum = np.mean(error_history)
        print 'Epoch: {}, error: {}.'.format(epoch+1,error_sum)
        #accuracy = np.mean(np.argmax(test_out,axis=1) == predict(test_in))
        #print '\tAccuracy: {} percent'.format(accuracy)


    print '\nprocess finished.\n'

if __name__ == '__main__':
    anncurvefit_train()
