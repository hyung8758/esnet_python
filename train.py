'''
Machine Learning Script Guide.
Please run this script to activate machine learning and try to
understand its structure and usage.
It is written based on tensorflow.

Support: DNN, RBM, RNN(LSTM)
                                                                    Written by Hyungwon Yang
                                                                                2017. 02. 26
                                                                                   EMCS Labs
'''

import main.loadfile as lf
import main.setvalues as set
import main.dnnnetworkmodels as net

# Setting initial parameters.
# Choose the problem type and continue the steps.

problem = 'classification' # classification, regression
fineTrainEpoch = 10
fineLearningRate = 0.01
learningRateDecay = 'off' # on, off
batchSize = 100
hiddenLayers = [100] # More than one hidden layer can be inserted. e.g.,[100,100]
hiddenFunction= 'sigmoid' # sigmoid, tanh
costFunction = 'adam' # gradient, adam
validationCheck = 'on' # If validationCheck is on, then 20% of train data will be taken for validation.
PlotGraph = 'off' # If this is on, graph will be saved in the dnn_graph directory.
                  # You can check the dnn structure on the tensorboard.
preTrainEpoch = 10 # rbm epoch.
preLearningRate = 0.01 # rbm learning rate.

######################################################################################
### RBM & DBN

# Load datasets.
print('Loading the data and setting default values...')
# # MNIST: classification
inputData, targetData, test_in, test_out = lf.readmnist()
# mfcc: regression
# inputData, targetData, test_in, test_out = lf.readartmfcc()

print('Setting default parameters...')
# Setting default values by using setvalues.
RBM_values = set.setParam(inputData=inputData,
                          targetData=targetData,
                          hiddenUnits=hiddenLayers
                          )

# Setting hidden layers: weightMatrix and biasMatrix
rbm_weightMatrix = RBM_values.genWeight()
rbm_biasMatrix = RBM_values.genBias()
rbm_input_x, rbm_input_y = RBM_values.genSymbol()

print('Constructing RBM model...')
rbm = net.RBMmodel(inputSymbol=rbm_input_x,
                   preTrainEpoch=preTrainEpoch,
                   preLearningRate=preLearningRate,
                   batchSize=batchSize,
                   weightMatrix=rbm_weightMatrix,
                   biasMatrix=rbm_biasMatrix,
                   )

rbm.genRBM()

print('RBM training...')
rbm.trainRBM(inputData)

# RBM has no test session. Save the trained variable and use it to DNN training session.
rbm_vars = rbm.getVariables()

rbm.closeRBM()

# Setting default values by using setvalues.
DNN_values = set.setParam(inputData=inputData,
                          targetData=targetData,
                          hiddenUnits=hiddenLayers
                          )
# Retrieve the weight and bias parameters from RBM training.
pretrained_weightMatrix = rbm_vars["weight"]
pretrained_biasMatrix = rbm_vars["bias"]

weightMatrix = DNN_values.setWeight(pretrained_weightMatrix)
biasMatrix = DNN_values.setBias(pretrained_biasMatrix)
input_x, input_y = DNN_values.genSymbol()

print('Constructing DNN model...')
dnn = net.DNNmodel(inputSymbol=input_x,
                   outputSymbol=input_y,
                   problem=problem,
                   fineTrainEpoch=fineTrainEpoch,
                   fineLearningRate=fineLearningRate,
                   learningRateDecay=learningRateDecay,
                   batchSize=batchSize,
                   hiddenFunction=hiddenFunction,
                   costFunction=costFunction,
                   validationCheck=validationCheck,
                   weightMatrix=weightMatrix,
                   biasMatrix=biasMatrix,
                   PlotGraph=PlotGraph
                   )

# Generate DNN network.
dnn.genDNN()

# Train DNN network.
dnn.trainDNN(inputData,targetData)

# Test trained DNN network.
dnn.testDNN(test_in,test_out)

vars = dnn.getVariables()

dnn.closeDNN()


########################################################################################################################
### DNN

import main.loadfile as lf
import main.setvalues as set
import main.dnnnetworkmodels as net

# Setting initial parameters.
# Choose the problem type and continue the steps.

# classification.
problem = 'classification' # classification, regression
fineTrainEpoch = 10
fineLearningRate = 0.01
learningRateDecay = 'off' # on, off
batchSize = 100
hiddenLayers = [100,100]
hiddenFunction= 'sigmoid'
costFunction = 'adam' # gradient, adam

print('Loading the data and setting default values...')
inputData, targetData, test_in, test_out = lf.readmnist()
# inputData, targetData, test_in, test_out = lf.readartmfcc()

print('Setting default parameters...')
# Setting default values by using SetValues.
DNN_values = set.setParam(inputData=inputData,
                          targetData=targetData,
                          hiddenUnits=hiddenLayers
                          )

# Setting hidden layers: weightMatrix and biasMatrix
weightMatrix = DNN_values.genWeight()
biasMatrix = DNN_values.genBias()
# Generating input symbols.
input_x, input_y = DNN_values.genSymbol()

print('Constructing DNN model...')
dnn = net.DNNmodel(inputSymbol=input_x,
                   outputSymbol=input_y,
                   problem=problem,
                   fineTrainEpoch=fineTrainEpoch,
                   fineLearningRate=fineLearningRate,
                   learningRateDecay=learningRateDecay,
                   batchSize=batchSize,
                   hiddenFunction=hiddenFunction,
                   costFunction=costFunction,
                   validationCheck=validationCheck,
                   weightMatrix=weightMatrix,
                   biasMatrix=biasMatrix,
                   PlotGraph=PlotGraph
                   )

# Generate DNN network.
dnn.genDNN()

# Train DNN network.
print('DNN training...')
dnn.trainDNN(inputData,targetData)

# Test trained DNN network.
dnn.testDNN(test_in,test_out)

# Save the variables.
vars = dnn.getVariables()
# Terminate
dnn.closeDNN()


########################################################################################################################
### CNN


########################################################################################################################
### RNN

### Simple LSTM
'''
Simple means it has only one hidden layer.
This "simpleRNNmodel" will be updated to "RNNmodel" when it is fixed to support multiple hidden layers.
'''
import _pickle as pickle
import numpy as np
import main.setvalues as set
import main.rnnnetworkmodels as net

# regression
problem = 'regression' # classification, regression
rnnCell = 'lstm' # rnn, lstm, gru
trainEpoch = 10
learningRate = 0.0001
learningRateDecay = 'off' # on, off
batchSize = 100
hiddenLayers = [200]
timeStep = 17
costFunction = 'adam' # gradient, adam
PlotGraph = 'off' # If this is on, graph will be saved in the rnn_graph directory.
                  # You can check the dnn structure on the tensorboard.

# Import data
# new_acoustics_vowel and new_articulation_vowel
with open("train_data/new_acoustics.pckl", "rb") as f:
    acoustics = pickle.load(f)
    # acoustics = dt.momentumSigmoid(acoustics,0.25)
    train_input = np.reshape(acoustics[0:18000],[18000,17,39])
    test_input = np.reshape(acoustics[18000:20000],[2000,17,39])
with open("train_data/new_articulation.pckl", "rb") as f:
    articulations = pickle.load(f)
    # articulations = dt.momentumSigmoid(articulations,0.1)
    train_output = np.reshape(articulations[0:18000],[18000,17,14])
    test_output = np.reshape(articulations[18000:20000],[2000,17,14])

# input should be three dimensional. [# of examples, # of timesteps, # of features]
lstm_values = set.simpleRNNParam(inputData=train_input,
                                 targetData=train_output,
                                 timeStep=timeStep,
                                 hiddenUnits=hiddenLayers
                                 )

# Setting hidden layers: weightMatrix and biasMatrix
lstm_weightMatrix = lstm_values.genWeight()
lstm_biasMatrix = lstm_values.genBias()
lstm_input_x,lstm_input_y = lstm_values.genSymbol()

lstm_net = net.simpleRNNModel(inputSymbol=lstm_input_x,
                              outputSymbol=lstm_input_y,
                              rnnCell=rnnCell,
                              problem=problem,
                              trainEpoch=trainEpoch,
                              learningRate=learningRate,
                              timeStep=timeStep,
                              batchSize=batchSize,
                              validationCheck=validationCheck,
                              weightMatrix=lstm_weightMatrix,
                              biasMatrix=lstm_biasMatrix)

# Generate a RNN(lstm) network.
lstm_net.genRNN()
# Train the RNN(lstm) network.
lstm_net.trainRNN(train_input,train_output)
# Test the trained RNN(lstm) network.
lstm_net.testRNN(test_input,test_output)
# Save the trained parameters.
vars = lstm_net.getVariables()
# Terminate the session.
lstm_net.closeRNN()


