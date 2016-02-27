# -*- coding: utf-8 -*-
"""
Deep Belief Network for Classification problem.

This script is consisted of 2 parts: pre-training and fine-tuning. In pre-trainng,
RBM is applied to train the hidden and bias units. This unsupervised technique
modified the weight and biases based on the probabilistic algorithm so that it shapes
the hidden layers to have desirable properties.
In Fine-tuning, the DNN is running with the hidden layer units that had been modified
in pre-training session. This step is just like ANN.
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
                                                                                2016. 02. 23
                                                                                    EMCS Lab
"""

import numpy as np
import theano
import loadfile
import errortools as et
import binarySigmoid as logistic
import random

from SetValues import SetDBN
from NetworkModel import DBN_model



def dbnclass_train():

    print 'Loading the data and setting default values...'

    # Import MNIST (optional)
    train_in, train_out, test_in, test_out = loadfile.readmnist()

    # Import cancerData
    #train_in, train_out, test_in, test_out = loadfile.readcancer()


    print 'Structuring the hidden layers...'

    # Setting default values by using  SetValues.
    hiddenStructure = np.array([50,50,50])
    values = SetDBN(inputs=train_in,
                    outputs=train_out,
                    learningRate=0.01,
                    momentum=0.9,
                    pre_epoch=1,
                    fine_epoch=1,
                    hiddenUnits=hiddenStructure,
                    W=None,
                    b=None)

    # Setting hidden layers: weightMatrix and biasMatrix
    weightMatrix = values.genWeight()
    biasMatrix = values.genBias()
    inX, outY = values.genMatrices()

    # the length of the hidden layer
    hidden_num = len(hiddenStructure)
    pre_epoch, fine_epoch, lr = values.epochs()
    momentum = 0.9


################################
#####   PRETRAIN SESSION   #####
################################

    print '\nPre_training the data...'

    inputPattern = train_in

    # The number of hidden layers for training
    for hidden in range(hidden_num):

        vhMatrix = weightMatrix[hidden]
        hBiasMatrix = biasMatrix[hidden+1]
        vBiasMatrix = biasMatrix[hidden]
        inputSave = []

        # training the data for given epochs
        for epoch in range(pre_epoch):

            if type(inputPattern) is not list:
                rand_num = np.random.permutation(len(inputPattern))
                layerForPT = inputPattern[rand_num]
            else:
                layerForPT = [ran for ran in inputPattern]
                random.shuffle(layerForPT)


            # training each data for updating weights and biases(unsupervised learning)
            for num in range(len(inputPattern)):

                ### visual0
                if type(layerForPT) is not list:
                    visual0Array = np.array([layerForPT[num]])
                else:
                    visual0Array = layerForPT[num]
                # hidden0
                hidden0 = np.dot(visual0Array,vhMatrix) + hBiasMatrix
                hidden0Array = logistic.binarySigmoid(momentum,hidden0)

                ### visual1
                visual1 = np.dot(hidden0Array,vhMatrix.T) + vBiasMatrix
                visual1Array = logistic.binarySigmoid(momentum,visual1)

                ### hidden1
                hidden1 = np.dot(visual1Array,vhMatrix) + hBiasMatrix
                hidden1Array = logistic.binarySigmoid(momentum,hidden1)

                # update weights and biases
                vhMatrix += lr * (np.dot(visual0Array.T,hidden0Array) - np.dot(visual1Array.T,hidden1Array))
                vBiasMatrix += lr * (visual0Array - visual1Array)
                hBiasMatrix += lr * (hidden0Array - hidden1Array)
                if epoch+1 == pre_epoch:

                    inputSave.append(hidden0Array)

            print '{}/{} Hidden Layer, {}/{} epoch'.format(hidden+1,hidden_num,epoch+1,pre_epoch)

        weightMatrix[hidden] = vhMatrix
        biasMatrix[hidden+1] = hBiasMatrix
        biasMatrix[hidden] = vBiasMatrix
        inputPattern = inputSave


    # Sharing the weight and bias
    weightMatrix, biasMatrix = values.sharing(weightMatrix,biasMatrix)


    print 'Pre_training completed. Prepare for fine_turning...'


################################
#####   FINETUNE SESSION   #####
################################

    # Fine turning
    print '\nFine_tuning the data...'

    total_epochs = fine_epoch
    print 'Constructing DBN_model...'
    models = DBN_model(inX=inX,
                   outY=outY,
                   weightMatrix=weightMatrix,
                   biasMatrix=biasMatrix,
                   learningRate=0.001)

    outputStorage, outputActivation = models.fnn(methods='curvefitting')
    cost, updates = models.sgd2(outputStorage)

    # Train and Predict function.
    train = theano.function(inputs=[inX,outY],outputs=cost,updates=updates,allow_input_downcast=True)
    predict = theano.function(inputs=[inX],outputs=outputActivation[-1],allow_input_downcast=True)


    print 'Training the data...'

    error_epoch = 10
    total_epoch = fine_epoch
    for epoch in range(total_epoch):

        # Shuffling inputs and outputs
        rand_num = np.random.permutation(len(train_in))
        train_in = train_in[rand_num]
        train_out = train_out[rand_num]
        error_history = []

        #total_inputs = range(train_in.shape[0])
        for batch_in,batch_out in zip(range(train_in.shape[0]),range(1,train_in.shape[0]+1)):

            #error = train(train_in[batch_in:batch_out],train_out[batch_in:batch_out])
            error = train(np.array([train_in[batch_in]]),np.array([train_out[batch_in]]))
            error_history.append(error)
        error_sum = np.mean(error_history)

        # Print error.
        if epoch%error_epoch == 0:

            print 'Epoch: {}, error: {}.'.format(epoch+1,error_sum)

    print '\nFine_tuning completed. Testing the result...'


###############################
#####   TESTING SESSION   #####
###############################

    # Testing the trained model and its accuracy.

    # accuracy bug needs to be solved.
    accuracy = np.mean(np.argmax(test_out,axis=1) == predict(test_in))
    print '\nTesting completed. Testing Result is as follows\nAccuracy: {} percent'.format(round(accuracy*100,2))

    # Confusion plot (update soon)


    print'ANN learning procedure has been completed.'


if __name__ == '__main__':
    dbnclass_train()



