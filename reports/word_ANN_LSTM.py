# Word prediction training with two methods: ANN and LSTM(RNN)

import re
import numpy as np
import main.setvalues as set
import main.networkmodels as net

# data preprocessing.
# import data.
with open('train_data/pg8800_train','r') as train_n:
    train_ngram = np.loadtxt(train_n.readlines(),dtype=int)

with open('train_data/pg8800_test','r') as test_n:
    test_ngram = np.loadtxt(test_n.readlines(),dtype=int)

with open('train_data/pg8800_words','r') as look_w:
    lookup = look_w.readlines()
    lookup_words = []
    for string in lookup:
        lookup_words.append(re.sub('\n','',string))

vocab_size = len(lookup_words)
train_data_size = len(train_ngram)
test_data_size = len(test_ngram)

# one hot coding.

# train_inputs
train_inputs = np.zeros((train_data_size,vocab_size))
train_outputs = np.zeros((train_data_size,vocab_size))
box = 0
for idx in train_ngram:
    train_inputs[box][idx[0:3]] = 1
    train_outputs[box][idx[-1]] = 1
    box += 1
# test_inputs
test_inputs = np.zeros((test_data_size,vocab_size))
test_outputs = np.zeros((test_data_size,vocab_size))
box = 0
for idx in test_ngram:
    test_inputs[box][idx[0:3]] = 1
    train_outputs[box][idx[-1]] = 1
    box += 1


### ANN
# parameter setting.

fineTrainEpoch = 100
fineLearningRate = 0.001
learningRateDecay = 'off' # on, off
batchSize = 100
hiddenLayers = [100]
problem = 'classification' # classification, regression
hiddenFunction= 'tanh'
costFunction = 'adam' # gradient, adam
PlotGraph = 'off'

DNN_values = set.setParam(inputData=train_inputs,
                    targetData=train_outputs,
                    hiddenUnits=hiddenLayers
                    )

# Setting hidden layers: weightMatrix and biasMatrix
weightMatrix = DNN_values.genWeight()
biasMatrix = DNN_values.genBias()
# Generating input symbols.
input_x, input_y = DNN_values.genSymbol()

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
dnn.trainDNN(train_inputs,train_outputs)

# Test trained DNN network.
dnn.testDNN(test_inputs,test_outputs)

vars = dnn.getVariables()

dnn.closeDNN()

######################################################################################
### LSTM
import re
import numpy as np
import main.setvalues as set
import main.networkmodels as net

# data preprocessing.
# import data.
with open('train_data/pg8800_train','r') as train_n:
    train_ngram = np.loadtxt(train_n.readlines(),dtype=int)

with open('train_data/pg8800_test','r') as test_n:
    test_ngram = np.loadtxt(test_n.readlines(),dtype=int)

with open('train_data/pg8800_words','r') as look_w:
    lookup = look_w.readlines()
    lookup_words = []
    for string in lookup:
        lookup_words.append(re.sub('\n','',string))

vocab_size = len(lookup_words)
train_data_size = len(train_ngram)
test_data_size = len(test_ngram)

# regression
trainEpoch = 10
learningRate = 0.001
learningRateDecay = 'off' # on, off
batchSize = 100
hiddenLayers = [100]
timeStep = 3
problem = 'classification' # classification, regression
costFunction = 'adam' # gradient, adam
plotOption = 'off'

# data preprocessing.
# dictionary list.
word_box = np.identity(len(lookup_words),dtype=int)
input_box = np.zeros((timeStep,vocab_size))
lstm_train_inputs = np.empty((1,timeStep,vocab_size))
lstm_train_outputs = np.empty((1,timeStep,vocab_size))
con=0
for idx in train_ngram:
    input_box = np.zeros((timeStep, vocab_size))
    output_box = np.zeros((timeStep, vocab_size))
    for input in list(range(timeStep)):
        input_box[input][idx[input]] = 1
    for output in list(range(timeStep)):
        output_box[output][idx[output+1]] = 1
    lstm_train_inputs = np.append(lstm_train_inputs,[input_box],axis=0)
    lstm_train_outputs = np.append(lstm_train_outputs, [output_box], axis=0)
    con +=1
    if con % 500 == 0:
        print('{} / {} is completed'.format(con,train_data_size))




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
                               trainEpoch=trainEpoch,
                               learningRate=learningRate,
                               timeStep=timeStep,
                               batchSize=batchSize,
                               weightMatrix=lstm_weightMatrix,
                               biasMatrix=lstm_biasMatrix)

lstm_net.genLSTM()

lstm_net.trainLSTM(lstm_train_inputs,lstm_train_outputs)

# lstm_net.testLSTM(lstm_test_inputs,lstm_test_outputs)

vars = lstm_net.getVariables()

lstm_net.closeLSTM()
