# -*- coding: utf-8 -*-
'''
Artificial Neural Network for Curvefitting problems (regression).
(Online learning)

This script is written for activating a simple artificial neural network. (ANN)
Please refer to the step as follows.

1. Choose the dataset and uncomment it in order to train the dataset.
   (Find more datasets in the loadfile.py script.)
2. Adjust parameters such as learningrate, momentum or hiddenunits.
   Manipulating the parameters is required to derive the best learning performance
   and it might be achieved when the combination of parameters reflects the characteristics
   of the datasets.
3. After training and testing sessions, MSE (Mean Square Error) and pearson r will be displayed.
   The visual plotting that explains the learning result will also help the interpretation.

                                                                    Written by Hyungwon Yang
                                                                                2016. 02. 10
                                                                                    EMCS Lab
'''

import numpy as np
import theano
import loadfile
import errortools as et

from SetValues import SetANN
from NetworkModel import ANN_model



def anncurvefit_train():

    print 'Loading the data and setting default values...'

    # Import mfcc data
    train_in, train_out, test_in, test_out = loadfile.readartmfcc()

    # Import bodyfat (optional)
    #train_in, train_out, test_in, test_out = loadfile.readbody()

    # Import building (optional)
    #train_in, train_out, test_in, test_out = loadfile.readbuilding()

    print 'Constructing the hidden layers...'

    # Generate initial values.
    values = SetANN(inputs=train_in,
                       outputs=train_out,
                       learningRate=0.001,
                       momentum=0.9,
                       epochNum=50,
                       hiddenUnits=10,
                       W=None,
                       b=None)
    ihMatrix,hoMatrix = values.genWeight()
    hBiasMatrix, oBiasMatrix = values.genBias()
    inX, outY = values.genMatrices()
    # error display rate
    error_epoch = 10 # 1 out of 10

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

    # Train and Predict function.
    train = theano.function(inputs=[inX,outY],outputs=cost,updates=updates,allow_input_downcast=True)
    predict = theano.function(inputs=[inX],outputs=outputActivation,allow_input_downcast=True)


################################
#####   TRAINING SESSION   #####
################################

    # Training the data
    print 'Training the data...'

    total_epoch = range(values.epochs())
    for epoch in total_epoch:

        error_history = []
        # Shuffling inputs and outputs
        rand_num = np.random.permutation(len(train_in))
        train_in = train_in[rand_num]
        train_out = train_out[rand_num]

        # online learning.
        for data_in,data_out in zip(range(train_in.shape[0]),range(1,train_in.shape[0]+1)):

            error = train(train_in[data_in:data_out],train_out[data_in:data_out])
            error_history.append(error)

        error_sum = np.mean(error_history)

        if epoch%error_epoch == 0:
            # print error rate and prediction correctness
            print 'Epoch: {}, error: {}.'.format(epoch+1,error_sum)

    print '\nprocess finished.\n\nTesting the result...'


###############################
#####   TESTING SESSION   #####
###############################

    # Testing the trained model.
    test_error_history =[]

    # Extract the results by processing test inputs.
    for data_test_in,data_test_out in zip(range(test_in.shape[0]),range(1,test_in.shape[0]+1)):

            test_error = predict(test_in[data_test_in:data_test_out])
            test_error_history.append(test_error)

    # Calculating MSE and correlation coefficients
    MSE = et.MSE(test_out,test_error_history)
    cor_coef = et.pearsonR(test_out,test_error_history)
    print 'Testing completed. Testing Result is as follows\nMSE          : {}\nCorrelation r: {}'.format(round(MSE,2),round(cor_coef,3))

    # Plotting correlation data
    print 'Displaying correlation graph...\n The plot will be displayed in a moment.'
    et.correlationplot(test_out,test_error_history)

    print'ANN learning procedure has been completed.'


if __name__ == '__main__':
    anncurvefit_train()
