# -*- coding: utf-8 -*-
'''
Artificial Neural Network for Classification problems.
(Online learning)

This script is written for activating a simple artificial neural network. (ANN)
Please refer to the step as follows.

1. Choose the dataset and uncomment it in order to train the dataset.
   (Find more datasets in the loadfile.py script.)
2. Adjust parameters such as learningrate, momentum or hiddenunits.
   Manipulating the parameters is required to derive the best learning performance
   and it might be achieved when the combination of parameters reflects the characteristics
   of the datasets.
3. After training and testing sessions, MSE (Mean Square Error) and prediction accuracy will be displayed.
   The visual plotting that explains the learning result will also help the interpretation.


                                                                    Written by Hyungwon Yang
                                                                                2016. 02. 07
                                                                                    EMCS Lab
'''

import numpy as np
import theano
import loadfile

from SetValues import SetANN
from NetworkModel import ANN_model



def annclass_train():

    print 'Loading the data and setting default values...'


    # Import MNIST (optional)
    train_in, train_out, test_in, test_out = loadfile.readmnist()

    # Import cancerData
    #train_in, train_out, test_in, test_out = loadfile.readcancer()

    print 'Constructing the hidden layers...'

    # Generate initial values.
    values = SetANN(inputs=train_in,
                       outputs=train_out,
                       learningRate=0.001,
                       momentum=0.9,
                       epochNum=30,
                       hiddenUnits=50,
                       W=None,
                       b=None)
    ihMatrix,hoMatrix = values.genWeight()
    hBiasMatrix, oBiasMatrix = values.genBias()
    inX, outY = values.genMatrices()

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


################################
#####   TRAINING SESSION   #####
################################

    # Training the data
    print 'Training the data...'

    total_epoch = range(values.epochs())
    error_history = []
    for epoch in total_epoch:

        # Shuffling inputs and outputs
        rand_num = np.random.permutation(len(train_in))
        train_in = train_in[rand_num]
        train_out = train_out[rand_num]

        #total_inputs = range(train_in.shape[0])
        for data_in,data_out in zip(range(train_in.shape[0]),range(1,train_in.shape[0]+1)):
        #for iter in total_inputs:
            error = train(train_in[data_in:data_out],train_out[data_in:data_out])
            error_history.append(error)

        # print error rate and prediction Accuracy
        error_sum = np.mean(error_history)
        print 'Epoch: {}, error: {}.'.format(epoch+1,error_sum)

###############################
#####   TESTING SESSION   #####
###############################

    # Testing the trained model and its accuracy.
    accuracy = np.mean(np.argmax(test_out,axis=1) == predict(test_in))
    print '\nTesting completed. Testing Result is as follows\nAccuracy: {} percent'.format(round(accuracy*100,2))

    print'ANN learning procedure has been completed.'

if __name__ == '__main__':
    annclass_train()