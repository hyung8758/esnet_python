'''
DNN(deep neural network) example.
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

# classification.
problem = 'regression'
fineTrainEpoch = 20
fineLearningRate = 0.001
learningRateDecay = 'off'
batchSize = 100
hiddenLayers = [300]
hiddenFunction= 'sigmoid'
costFunction = 'adam'
validationCheck = 'off'
plotGraph = 'off'


print('Loading the data and setting default values...')
# Data for classification.
inputData, targetData, test_in, test_out = lf.readmnist()


print('Setting default parameters...')
# Setting default values by using SetValues.
DNN_values = set.setParam(inputData=targetData,
                          targetData=inputData,
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
dnn.trainDNN(targetData,inputData)
# Test trained DNN network.
dnn.testDNN(test_out,test_in)
# Save the variables.
vars = dnn.getVariables()
# Terminate the session.
dnn.closeDNN()