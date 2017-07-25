'''
DBN(Deep Belief Network) and DNN(deep neural network) example.
Machine Learning Script Example.
Please run this script to activate machine learning and try to
understand its structure and usage.
It is written based on tensorflow.

                                                                               Hyungwon Yang
                                                                                2017. 02. 26
                                                                                   EMCS Labs
'''


import src.loadfile as lf
import src.setvalues as set
import src.dnnnetworkmodels as net

# Setting initial parameters.
# Choose the problem type and continue the steps.

problem = 'classification'
fineTrainEpoch = 10
fineLearningRate = 0.01
learningRateDecay = 'off'
batchSize = 100
hiddenLayers = [100]
hiddenFunction= 'sigmoid'
costFunction = 'adam'
validationCheck = 'off'
plotGraph = 'off'

preTrainEpoch = 10
preLearningRate = 0.01


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
