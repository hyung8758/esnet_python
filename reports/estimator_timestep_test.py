import _pickle as pickle
import numpy as np
import main.setvalues as set
import main.rnnnetworkmodels as net

# import data.
with open("/Users/hyungwonyang/Documents/data/5.python_data/HY_python_NN_data/train_data/acoustics_vowel.pckl","rb") as f:
    acoustics_data = pickle.load(f)

with open("/Users/hyungwonyang/Documents/data/5.python_data/HY_python_NN_data/train_data/articulation_vowel.pckl","rb") as f:
    articulation_data = pickle.load(f)

# data length check.
# data_cut=7
# timeLen=[]
# count=0
# for i in range(len(acoustics_data)):
#     timeLen.append(len(acoustics_data[i][0]))
#     if len(acoustics_data[i][0]) >= data_cut:
#         count += 1
# print("Total data: {}".format(len(acoustics_data)))
# print("When data cut is: {}, remained data is {} / {}".format(data_cut,count,len(acoustics_data)))
# loss = len(acoustics_data)-count
# print("Lost data: {}".format(loss))

# data control parameters
data_cut = 7

# generate data inputs(acoustics) and outputs(articulations)
data_input = []
data_output = []
count = 0

for i in range(len(acoustics_data)):
    if len(acoustics_data[i][0]) >= data_cut:
        count += 1
        data_input.append(acoustics_data[i])
        data_output.append(articulation_data[i])
print("Total data: {}".format(len(acoustics_data)))
print("When data cut is: {}, remained data is {} / {}".format(data_cut, count, len(acoustics_data)))
loss = len(acoustics_data) - count
print("Lost data: {}".format(loss))

# set data squeeze function.
def sigmoid(input):
    return 1/ (1 + np.exp(-0.25 * input))


train_input_data = []
train_output_data = []
test_input_data = []
test_output_data = []
for turn in range(20000):
    train_input_data.append(sigmoid(data_input[turn]))
    train_output_data.append(sigmoid(data_output[turn]))
for turn in range(4000):
    test_input_data.append(sigmoid(data_input[turn+20000]))
    test_output_data.append(sigmoid(data_output[turn+20000]))

# Generate dataset.
def ann2rnn(input_data,output_data,timeStep):

    data_size = len(input_data)
    if len(input_data) != len(output_data):
        raise ValueError("The number of input and output data is different.")

    input_feat_size = input_data[0].shape[0]
    output_feat_size = output_data[0].shape[0]

    # Generate dataset.
    tmp_input = np.zeros((1, timeStep, input_feat_size))
    tmp_output = np.zeros((1, timeStep, output_feat_size))
    print("timeStep is {}".format(timeStep))
    input_box = np.zeros((1, timeStep , input_feat_size))
    output_box = np.zeros((1, timeStep , output_feat_size))
    for dat in range(data_size):
        if dat % 1000 == 0:
            print("Current data processing: {} / {} ".format(dat,data_size))
        each_length = input_data[dat][0].size
        seq = int(each_length/timeStep)
        if seq == 0:
            continue
        else:
            inc_val=0
            for val in range(seq):
                for times in range(timeStep):
                    tmp_input[0][times] = input_data[dat][:,inc_val]
                    input_box = np.vstack((input_box, tmp_input))

                    tmp_output[0][times] = output_data[dat][:, inc_val]
                    output_box = np.vstack((output_box, tmp_output))
                    inc_val += 1

    input_data_shape = input_box.shape
    output_data_shape = output_box.shape
    print('RNN format data (timeStep: {}) is successfully built.'.format(timeStep))
    print("Input dataset dimension: examples: {}, timeStep: {}, features: {}".format(
                                                input_data_shape[0], input_data_shape[1], input_data_shape[2]))
    print("Output dataset dimension: examples: {}, timeStep: {}, features: {}".format(
                                                output_data_shape[0], output_data_shape[1], output_data_shape[2]))
    onehot_data = (input_box, output_box)

    return onehot_data


for cell in 'rnn','lstm':
    for timeStep in range(1,8):
        ### RNN test
        # parameters
        problem = 'regression'
        rnnCell = cell
        trainEpoch = 200
        learningRate = 0.00001
        learningRateDecay = 'off'
        batchSize = 100
        dropout = 'off'  # on, off
        hiddenLayers = [200]
        timeStep = timeStep
        costFunction = 'adam'  # gradient, adam
        validationCheck = 'off'  # if validationCheck is on, then 20% of train data will be taken for validation.
        plotGraph = 'off'  # If this is on, graph will be saved in the rnn_graph directory.
        # You can check the dnn structure on the tensorboard.

        train_input, train_output = ann2rnn(train_input_data, train_output_data, timeStep)
        test_input, test_output = ann2rnn(test_input_data, test_output_data, timeStep)


        rnn_values = set.RNNParam(inputData=train_input,
                                  targetData=train_output,
                                  timeStep=timeStep,
                                  hiddenUnits=hiddenLayers)

        # Setting hidden layers: weightMatrix and biasMatrix
        rnn_weightMatrix = rnn_values.genWeight()
        rnn_biasMatrix = rnn_values.genBias()
        rnn_input_x, rnn_input_y = rnn_values.genSymbol()

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

        # Generate a RNN(lstm) network.
        rnn_net.genRNN()
        # Train the RNN(lstm) network.
        rnn_net.trainRNN(train_input, train_output)
        # Test the trained RNN(lstm) network.
        rnn_net.testRNN(test_input, test_output)
        # Save the trained parameters.
        vars = rnn_net.getVariables()
        # Terminatelstm_net.closeRNN() the session.
        rnn_net.closeRNN()
