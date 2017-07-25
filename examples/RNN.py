'''
RNN(recurrent neural network) example....
Machine Learning Script Example.
Please run this script to activate machine learning and try to
understand its structure and usage.
It is written based on tensorflow.

                                                                               Hyungwon Yang
                                                                                2017. 02. 26
                                                                                   EMCS Labs
'''

import src.setvalues as set
import src.rnnnetworkmodels as net
import src.loadfile as lf

# regression
problem = 'regression'
rnnCell = 'lstm'
trainEpoch = 3
learningRate = 0.001
learningRateDecay = 'off'
batchSize = 100
dropout = 'off'
hiddenLayers = [200]
timeStep = 17
costFunction = 'adam'
validationCheck = 'off'
plotGraph = 'off'

# Import data
# new_acoustics_vowel and new_articulation_vowel
train_input, train_output, test_input, test_output = lf.readartandacou()

# input should be three dimensional. [# of examples, # of timesteps, # of features]
rnn_values = set.RNNParam(inputData=train_input,
                          targetData=train_output,
                          timeStep=timeStep,
                          hiddenUnits=hiddenLayers)

# Setting hidden layers: weightMatrix and biasMatrix
rnn_weightMatrix = rnn_values.genWeight()
rnn_biasMatrix = rnn_values.genBias()
rnn_input_x,rnn_input_y = rnn_values.genSymbol()

rnn_net = net.RNNModel(inputSymbol=rnn_input_x,
                       outputSymbol=rnn_input_y,
                       rnnCell=rnnCell,
                       problem=problem,
                       hiddenLayer=hiddenLayers,
                       trainEpoch=trainEpoch,
                       learningRate=learningRate,
                       learningRateDecay=learningRateDecay,
                       timeStep=timeStep,
                       batchSize=batchSize,
                       dropout=dropout,
                       validationCheck=validationCheck,
                       plotGraph=plotGraph,
                       weightMatrix=rnn_weightMatrix,
                       biasMatrix=rnn_biasMatrix)

# Generate a RNN network.
rnn_net.genRNN()
# Train the RNN network.
rnn_net.trainRNN(train_input,train_output)
# Test the trained RNN network.
rnn_net.testRNN(test_input,test_output)
# Save the trained parameters.
vars = rnn_net.getVariables()
# Terminate the session.
rnn_net.closeRNN()
