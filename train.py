'''
Machine Learning Script Guide.
Please run this script to activate machine learning and try to
understand its structure and usage.
It is written based on tensorflow.

Support: DNN, RBM, RNN(LSTM)

Tutorial contents.
(1) RBM and DNN training.
(2) DNN training.
(3) CNN training. (not yet supported)
(4) RNN training.

                                                                               Hyungwon Yang
                                                                                2017. 02. 26
                                                                                   EMCS Labs
'''


######################################################################################
### (1) RBM & DNN
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
validationCheck = 'off' # If validationCheck is on, then 20% of train data will be taken for validation.
plotGraph = 'off' # If this is on, graph will be saved in the dnn_graph directory.
                  # You can check the dnn structure on the tensorboard.
preTrainEpoch = 10 # rbm epoch.
preLearningRate = 0.01 # rbm learning rate.


# Load datasets.
print('Loading the data and setting default values...')
# MNIST: classification
inputData, targetData, test_in, test_out = lf.readmnist()
# mfcc: regression
# inputData, targetData, test_in, test_out = lf.readartmfcc()

print('Setting default parameters...')
# Setting default values by using setvalues.
RBM_values = set.setParam(inputData=inputData,
                          targetData=targetData,
                          hiddenUnits=hiddenLayers)

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
                   biasMatrix=rbm_biasMatrix)

# Generate RBM network.
rbm.genRBM()
# Train RBM network.
rbm.trainRBM(inputData)
# RBM has no test session. Save the trained variable and use it to DNN training session.
rbm_vars = rbm.getVariables()
# Terminate the session.
rbm.closeRBM()


# Setting default values by using setvalues.
DNN_values = set.setParam(inputData=inputData,
                          targetData=targetData,
                          hiddenUnits=hiddenLayers)

# Retrieve the weight and bias parameters from RBM training.
pretrained_weightMatrix = rbm_vars["weight"]
pretrained_biasMatrix = rbm_vars["bias"]

# Use pretrained weight and bias for DNN training.
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
                   plotGraph=plotGraph,
                   weightMatrix=weightMatrix,
                   biasMatrix=biasMatrix)

# Generate DNN network.
dnn.genDNN()
# Train DNN network.
dnn.trainDNN(inputData,targetData)
# Test trained DNN network.
dnn.testDNN(test_in,test_out)
# Save the variables.
vars = dnn.getVariables()
# Terminate the session.
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
fineLearningRate = 0.001
learningRateDecay = 'off' # on, off
batchSize = 100
hiddenLayers = [100,100]
hiddenFunction= 'sigmoid' # sigmoid, tanh
costFunction = 'adam' # gradient, adam
validationCheck = 'on' # if validationCheck is on, then 20% of train data will be taken for validation.
PlotGraph = 'off' # If this is on, graph will be saved in the rnn_graph directory.
                  # You can check the dnn structure on the tensorboard.

print('Loading the data and setting default values...')
# Data for classification.
inputData, targetData, test_in, test_out = lf.readmnist()
# Data for regression.
# inputData, targetData, test_in, test_out = lf.readartmfcc()

print('Setting default parameters...')
# Setting default values by using SetValues.
DNN_values = set.setParam(inputData=inputData,
                          targetData=targetData,
                          hiddenUnits=hiddenLayers)

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
                   plotGraph=plotGraph,
                   weightMatrix=weightMatrix,
                   biasMatrix=biasMatrix)

# Generate DNN network.
dnn.genDNN()
# Train DNN network.
dnn.trainDNN(inputData,targetData)
# Test trained DNN network.
dnn.testDNN(test_in,test_out)
# Save the variables.
vars = dnn.getVariables()
# Terminate the session.
dnn.closeDNN()


########################################################################################################################
### CNN


########################################################################################################################
### RNN(basic, lstm, gru)

import main.setvalues as set
import main.rnnnetworkmodels as net
import main.loadfile as lf

# regression
problem = 'regression' # classification, regression
rnnCell = 'lstm' # rnn, lstm, gru
trainEpoch = 3
learningRate = 0.001
learningRateDecay = 'off' # on, off
batchSize = 100
dropout = 'off' # on, off
hiddenLayers = [200]
timeStep = 17
costFunction = 'adam' # gradient, adam
validationCheck = 'off' # if validationCheck is on, then 20% of train data will be taken for validation.
plotGraph = 'off' # If this is on, graph will be saved in the rnn_graph directory.
                  # You can check the network structure on the tensorboard.

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


