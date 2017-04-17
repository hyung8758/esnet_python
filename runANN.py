'''
This script is machine learning tutorial.
Please run this script to activate machine learning and try to
understand its structure and usage.
It is written based on tensorflow.

Support: DNN, RBM, LSTM
                                                                    Written by Hyungwon Yang
                                                                                2017. 02. 26
                                                                                   EMCS Labs
'''

import main.loadfile as lf
import numpy as np
import tensorflow as tf
import main.visualtools as vt
import main.setvalues as set
import main.networkmodels as net
import main.binarySigmoid as bs

# Setting initial parameters.
# Choose the problem type and continue the steps.

problem = 'classification' # classification, regression
fineTrainEpoch = 10
fineLearningRate = 0.01
learningRateDecay = 'off' # on, off
batchSize = 100
hiddenLayers = [100,100] # More than one hidden layer can be inserted.
hiddenFunction= 'sigmoid' # sigmoid, tanh
costFunction = 'adam' # gradient, adam
PlotGraph = 'on' # if this is on, graph will be saved on the dnn_graph directory.
preTrainEpoch = 10 # rbm epoch.
preLearningRate = 0.01 # rbm learning rate.

########################################################################################################################
### RBM & DBN
# Load datasets.

print('Loading the data and setting default values...')
# # MNIST: classification
inputData, targetData, test_in, test_out = lf.readmnist()
# mfcc: regression
# inputData, targetData, test_in, test_out = lf.readartmfcc()

print('Setting default parameters...')
# Setting default values by using SetValues.
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

rbm_vars = rbm.getVariables()

rbm.closeRBM()

# Setting default values by using SetValues.
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
                   weightMatrix=weightMatrix,
                   biasMatrix=biasMatrix
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
import main.networkmodels as net

# Setting initial parameters.
# Choose the problem type and continue the steps.

# classification.
training = 'on'
testing = 'on'
fineTrainEpoch = 10
fineLearningRate = 0.01
learningRateDecay = 'off' # on, off
batchSize = 100
normalize = 'off'
hiddenLayers = [100,100]
problem = 'classification' # classification, regression
hiddenFunction= 'sigmoid'
costFunction = 'adam' # gradient, adam
# plotOption = 'off' # This is not supported yet.

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
                   weightMatrix=weightMatrix,
                   biasMatrix=biasMatrix
                   )

# Generate DNN network.
dnn.genDNN()

# Train DNN network.
print('DNN training...')
dnn.trainDNN(inputData,targetData)

# Test trained DNN network.
dnn.testDNN(test_in,test_out)

vars = dnn.getVariables()

dnn.closeDNN()


########################################################################################################################
### CNN


########################################################################################################################
### RNN


########################################################################################################################
### LSTM

### Simple LSTM
import _pickle as pickle
import numpy as np
import main.setvalues as set
import main.datatool as dt
import main.lstmnetworkmodels as net

# regression
trainEpoch = 1
learningRate = 0.0001
learningRateDecay = 'off' # on, off
batchSize = 100
normalize = 'off'
hiddenLayers = [200]
hiddenNumber = 1
timeStep = 17
problem = 'regression' # classification, regression
costFunction = 'adam' # gradient, adam
plotOption = 'off'

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

# with open("train_data/acoustics.pckl", "rb") as f:
#     acoustics = pickle.load(f)
# with open("train_data/articulation.pckl", "rb") as f:
#     articulations = pickle.load(f)

# input should be three dimensional. [# of examples, # of timesteps, # of features]
lstm_values = set.simpleLSTMParam(inputData=train_input,
                           targetData=train_output,
                           timeStep=timeStep,
                           hiddenUnits=hiddenLayers
                           )

# Setting hidden layers: weightMatrix and biasMatrix
lstm_weightMatrix = lstm_values.genWeight()
lstm_biasMatrix = lstm_values.genBias()
lstm_input_x,lstm_input_y = lstm_values.genSymbol()

lstm_net = net.simpleLSTMmodel(inputSymbol=lstm_input_x,
                               outputSymbol=lstm_input_y,
                               problem=problem,
                               trainEpoch=trainEpoch,
                               learningRate=learningRate,
                               timeStep=timeStep,
                               batchSize=batchSize,
                               weightMatrix=lstm_weightMatrix,
                               biasMatrix=lstm_biasMatrix)

lstm_net.genLSTM()

lstm_net.trainLSTM(train_input,train_output)

lstm_net.testLSTM(test_input,test_output)

vars = lstm_net.getVariables()

lstm_net.closeLSTM()


'''
### LSTM
import _pickle as pickle
import numpy as np
import main.setvalues as set
import main.datatool as dt
import main.lstmnetworkmodels as net

# regression
trainEpoch = 50
learningRate = 0.0001
learningRateDecay = 'off' # on, off
batchSize = 10
normalize = 'off'
hiddenLayers = [200]
hiddenNumber = 1
timeStep = 17
problem = 'regression' # classification, regression
costFunction = 'adam' # gradient, adam
plotOption = 'off'


# Import data
with open("train_data/new_acoustics_vowel.pckl", "rb") as f:
    acoustics = pickle.load(f)
    acoustics = dt.momentumSigmoid(acoustics, 0.25)
with open("train_data/new_articulations_vowel.pckl", "rb") as f:
    articulations = pickle.load(f)
    articulations = dt.momentumSigmoid(articulations,0.1)


# Get lstm parameters.
lstm_values = set.simpleLSTMParam(inputData=acoustics,
                           targetData=articulations,
                           timeStep=timeStep,
                           hiddenUnits=hiddenLayers
                           )

# Setting hidden layers: weightMatrix and biasMatrix
lstm_weightMatrix = lstm_values.genWeight()
lstm_biasMatrix = lstm_values.genBias()
lstm_input_x,lstm_input_y = lstm_values.genSymbol()


lstm_net = net.simpleLSTMmodel(inputSymbol=lstm_input_x,
                               outputSymbol=lstm_input_y,
                               trainEpoch=trainEpoch,
                               learningRate=learningRate,
                               timeStep=timeStep,
                               batchSize=batchSize,
                               weightMatrix=lstm_weightMatrix,
                               biasMatrix=lstm_biasMatrix)

lstm_net.genLSTM()

lstm_net.trainLSTM(acoustics,articulations)

lstm_net.testLSTM()

lstm_net.closeLSTM()
'''


########################################################################################################################
### GRU

