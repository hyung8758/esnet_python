# -*- coding: utf-8 -*-
"""
Just note for testing...
Please skip this script.

@author: hyungwonyang
"""
'''
class jeagu():
    
    def respect(self):
        print '존경해요~'


class sunghah(jeagu):
    
    def jodenge(self):
        print 'jojo~'


def nicesum(A):
    B = []
    for i in range(1,len(A)+1):
        B.append(i**i)
    C = sum(B)
    return C


# For checking GPU or CPU usage

from theano import function, config, shared, sandbox
import theano.tensor as T
import numpy
import time

vlen = 10 * 30 * 768  # 10 x #cores x # threads per core
iters = 1000

rng = numpy.random.RandomState(22)
x = shared(numpy.asarray(rng.rand(vlen), config.floatX))
f = function([], T.exp(x))
print(f.maker.fgraph.toposort())
t0 = time.time()
for i in xrange(iters):
    r = f()
t1 = time.time()
print("Looping %d times took %f seconds" % (iters, t1 - t0))
print("Result is %s" % (r,))
if numpy.any([isinstance(x.op, T.Elemwise) for x in f.maker.fgraph.toposort()]):
    print('Used the cpu')
else:
    print('Used the gpu')

A_number = [63,234,76,43,23,88,55,46,16,88,245]
number_box = []
number_list = [1,4,5]
for check in number_list:
    number_box.append(A_number[check])



import random
intGen = []
for numbers in range(10):
    intGen.append(random.random())
    print '%d cost: ' % numbers, intGen[numbers]

fileName = 'Obama_Address.txt'
text = open(fileName,'r')
data = text.readlines()
allwords = []

for line in data:
    toss = line.split()
    for putin in toss:
        allwords.append(putin)


class Sum(object):

    def summation(self,x,y):
        z = x+y
        return z

class Div(object):
    def division(self,x,y):
        z = x/y
        return z


class Together(Sum,Div):
    def multiplication(self,x,y):
        z = x*y
        return z

class Numbers():

    def justsum(self,a,b):
        a = 10
        c = a+b
        return c
    def selfsum(self,a,b):
        self.a = 10
        c = self.a+b
        return c


class Walk():
    def setValue(self,personName,oneStep):
	    self.name = personName
	    self.step = oneStep
    def shortwalk(self):
	    print 'my name is %s' % self.name
	    return self.step * 3
    def longwalk(self):
	    print 'my name is %s' % self.name
	    return self.step * 7


class Gretting():
    def __init__(self):
        print 'welcome to python'
        self.good = 'good afternoon'
    def writing(self):
        print self.good


# '/Users/hyungwonyang/Google_Drive/Python/ML_sean'
### playing sound ###
import pyglet
song = pyglet.media.load('thesong.ogg')
song.play()


##
import time, wave, pymedia.audio.sound as sound
f= wave.open( 'YOUR FILE NAME', 'rb' )
sampleRate= f.getframerate()
channels= f.getnchannels()

format= sound.AFMT_S16_LE

snd= sound.Output( sampleRate, channels, format )
s= f.readframes( 300000 )
snd.play( s )

while snd.isPlaying():
    time.sleep( 0.05 )

### playing sound

import pygame

import time

pygame.init()

pygame.mixer.music.load("test.wav")

pygame.mixer.music.play()

time.sleep(10)

### resampling ###
import time, numpy, pygame.mixer, pygame.sndarray
from scikits.samplerate import resample

pygame.mixer.init(44100,-16,2,4096)

# choose a file and make a sound object
sound_file = "tone.wav"
sound = pygame.mixer.Sound(sound_file)

# load the sound into an array
snd_array = pygame.sndarray.array(sound)

# resample. args: (target array, ratio, mode), outputs ratio * target array.
# this outputs a bunch of garbage and I don't know why.
snd_resample = resample(snd_array, 1.5, "sinc_fastest")

# take the resampled array, make it an object and stop playing after 2 seconds.
snd_out = pygame.sndarray.make_sound(snd_resample)
snd_out.play()
time.sleep(2)



import numpy as np
import matplotlib.pyplot as plt

Age = [15, 16, 17, 18, 19, 20]
Tim = [150, 157, 164, 170, 178, 182]
Ninna = [145, 150, 152, 155, 165, 171]

xtotal = np.arange(len(Age))
width = 0.3

bar1 = plt.bar(xtotal, Tim, width, color='b')
bar2 = plt.bar(xtotal+width, Ninna, width, color='r')

plt.ylabel('Height')
plt.xlabel('Age')
plt.title('Height of Time and Ninna by their age')
plt.xticks(xtotal + width, Age)


from pylab import *
from numpy import *
from mpl_toolkits.mplot3d import axes3d

fig = figure()
ax = fig.gca(projection='3d')

# plot points in 3D
class1 = 0.6 * random.standard_normal((200,3))
ax.plot(class1[:,0],class1[:,1],class1[:,2],'o')
class2 = 1.2 * random.standard_normal((200,3)) + array([5,4,0])
ax.plot(class2[:,0],class2[:,1],class2[:,2],'o')
class3 = 0.3 * random.standard_normal((200,3)) + array([0,3,2])
ax.plot(class3[:,0],class3[:,1],class3[:,2],'o')



import numpy as np
import matplotlib.pyplot as plt


N = 150
r = 2 * np.random.rand(N)
theta = 2 * np.pi * np.random.rand(N)
area = 200 * r**2 * np.random.rand(N)
colors = theta

ax = plt.subplot(111, projection='polar')
c = plt.scatter(theta, r, c=colors, s=area, cmap=plt.cm.hsv)
c.set_alpha(0.75)

plt.show()

import numpy as np


class PointBrowser(object):
    """
    Click on a point to select and highlight it -- the data that
    generated the point will be shown in the lower axes.  Use the 'n'
    and 'p' keys to browse through the next and previous points
    """

    def __init__(self):
        self.lastind = 0

        self.text = ax.text(0.05, 0.95, 'selected: none',
                            transform=ax.transAxes, va='top')
        self.selected, = ax.plot([xs[0]], [ys[0]], 'o', ms=12, alpha=0.4,
                                 color='yellow', visible=False)

    def onpress(self, event):
        if self.lastind is None:
            return
        if event.key not in ('n', 'p'):
            return
        if event.key == 'n':
            inc = 1
        else:
            inc = -1

        self.lastind += inc
        self.lastind = np.clip(self.lastind, 0, len(xs) - 1)
        self.update()

    def onpick(self, event):

        if event.artist != line:
            return True

        N = len(event.ind)
        if not N:
            return True

        # the click locations
        x = event.mouseevent.xdata
        y = event.mouseevent.ydata

        distances = np.hypot(x - xs[event.ind], y - ys[event.ind])
        indmin = distances.argmin()
        dataind = event.ind[indmin]

        self.lastind = dataind
        self.update()

    def update(self):
        if self.lastind is None:
            return

        dataind = self.lastind

        ax2.cla()
        ax2.plot(X[dataind])

        ax2.text(0.05, 0.9, 'mu=%1.3f\nsigma=%1.3f' % (xs[dataind], ys[dataind]),
                 transform=ax2.transAxes, va='top')
        ax2.set_ylim(-0.5, 1.5)
        self.selected.set_visible(True)
        self.selected.set_data(xs[dataind], ys[dataind])

        self.text.set_text('selected: %d' % dataind)
        fig.canvas.draw()


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    X = np.random.rand(100, 200)
    xs = np.mean(X, axis=1)
    ys = np.std(X, axis=1)

    fig, (ax, ax2) = plt.subplots(2, 1)
    ax.set_title('click on point to plot time series')
    line, = ax.plot(xs, ys, 'o', picker=5)  # 5 points tolerance

    browser = PointBrowser()

    fig.canvas.mpl_connect('pick_event', browser.onpick)
    fig.canvas.mpl_connect('key_press_event', browser.onpress)

    plt.show()

import numpy as np
from binarySigmoid import binarySigmoid
from SetValues import shakeBatch

hiddenNumber=3
inputPattern = inputData
# The number of hidden layers for training
for hidden in range(hiddenNumber):

    # Assign input data, weight and bias
    vhMatrix = weightMatrix[hidden]
    hBiasMatrix = biasMatrix[hidden+1]
    vBiasMatrix = biasMatrix[hidden]
    inputSave = []

    # training the data for given epochs
    for epoch in range(preTrainEpoch):

        # Randomize all the training data.
        layerForPT = inputPattern
        batchIndex = shakeBatch(inputNumber,batchSize,'train')

        # training each data for updating weights and biases(unsupervised learning)
        for num in range(len(batchIndex)-1):

            # Bias replication.
            batchInputNumber = len(batchIndex[num])
            batch_hBiasMatrix = np.tile(hBiasMatrix,[batchInputNumber,1])
            batch_vBiasMatrix = np.tile(vBiasMatrix,[batchInputNumber,1])

            ### visual0
            visual0Array = layerForPT[batchIndex[num]]

            # hidden0
            hidden0 = np.dot(visual0Array,vhMatrix) + batch_hBiasMatrix
            hidden0Array = binarySigmoid(momentum,hidden0)

            ### visual1
            visual1 = np.dot(hidden0Array,vhMatrix.T) + batch_vBiasMatrix
            visual1Array = binarySigmoid(momentum,visual1)

            ### hidden1
            hidden1 = np.dot(visual1Array,vhMatrix) + batch_hBiasMatrix
            hidden1Array = binarySigmoid(momentum,hidden1)

            # update weights and biases
            vhMatrix += prelr * (np.dot(visual0Array.T,hidden0Array) - np.dot(visual1Array.T,hidden1Array))
            vBiasMatrix += np.mean(prelr * (visual0Array - visual1Array),axis=0)
            hBiasMatrix += np.mean(prelr * (hidden0Array - hidden1Array),axis=0)
            if epoch+1 == preTrainEpoch:

                for line in hidden0Array:
                    inputSave.append(line)

        print '{}/{} Hidden Layer, {}/{} epoch'.format(hidden+1,hiddenNumber,epoch+1,preTrainEpoch)

    weightMatrix[hidden] = vhMatrix
    biasMatrix[hidden+1] = hBiasMatrix
    biasMatrix[hidden] = vBiasMatrix
    if epoch+1 == preTrainEpoch:
        inputPattern = np.asarray(inputSave)



        for epoch in range(preTrainEpoch):

                # Randomize all the training data.
                layerForPT = inputPattern
                batchIndex = shakeBatch(inputNumber,batchSize,'train')

                # training each data for updating weights and biases(unsupervised learning)
                for num in range(len(batchIndex)-1):

                    # Bias replication.
                    batchInputNumber = len(batchIndex[num])
                    batch_hBiasMatrix = np.tile(hBiasMatrix,[batchInputNumber,1])
                    batch_vBiasMatrix = np.tile(vBiasMatrix,[batchInputNumber,1])

                    ### visual0
                    visual0Array = layerForPT[batchIndex[num]]

                    # hidden0
                    hidden0 = np.dot(visual0Array,vhMatrix) + batch_hBiasMatrix
                    hidden0Array = binarySigmoid(momentum,hidden0)

                    ### visual1
                    visual1 = np.dot(hidden0Array,vhMatrix.T) + batch_vBiasMatrix
                    visual1Array = binarySigmoid(momentum,visual1)

                    ### hidden1
                    hidden1 = np.dot(visual1Array,vhMatrix) + batch_hBiasMatrix
                    hidden1Array = binarySigmoid(momentum,hidden1)

                    # update weights and biases
                    vhMatrix += prelr * (np.dot(visual0Array.T,hidden0Array) - np.dot(visual1Array.T,hidden1Array))
                    vBiasMatrix += np.mean(prelr * (visual0Array - visual1Array),axis=0)
                    hBiasMatrix += np.mean(prelr * (hidden0Array - hidden1Array),axis=0)
                    if epoch+1 == preTrainEpoch:

                        for line in hidden0Array:
                            inputSave.append(line)

                print '{}/{} Hidden Layer, {}/{} epoch'.format(hidden+1,hiddenNumber,epoch+1,preTrainEpoch)

            weightMatrix[hidden] = vhMatrix
            biasMatrix[hidden+1] = hBiasMatrix
            biasMatrix[hidden] = vBiasMatrix
            if epoch+1 == preTrainEpoch:
                inputPattern = np.asarray(inputSave)

def unpickle(file):
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict
###
import numpy as np
from scipy.optimize import minimize

X = np.array([1,2,3,4,5,6,7,8,9,10])
Y = np.array([3,6,9,12,15,18,21,24,27,30])
W = np.random.rand(2)
equation = lambda X: np.array([X**0,X**1])
optim_equ = lambda W: 1/2.*sum((np.dot(equation(X).transpose(),W) - Y))**2

res = minimize(optim_equ,W,method='CG',options={'disp':True})
optimal_W = res.x




# Download MNIST dataset.
import loadfile

# mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

### Train
# import tensorflow as named tf for easy usage.
import tensorflow as tf
import main.loadfile as lf
import main.SetValues as sv
import numpy as np

train_in, train_out, test_in, test_out = lf.readmnist()
# Set x as an input variable. None means it can have any dimension vector.
x = tf.placeholder(tf.float32, [None, 784])

# Set weight and bias
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# input to output.
y = tf.nn.softmax(tf.matmul(x, W) + b)

# Set cross entropy.
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

# Set the train procedure.
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# ready to train.
init = tf.initialize_all_variables()

# Run the train procedure.
sess = tf.Session()
sess.run(init)

batchSize=100
start=0
last=1
for i in range(100):

    # for start, last in zip(range(0, len(test_in), batchSize), range(100, len(test_out), batchSize)):

    batch_xs = train_in[start:last*100]
    batch_ys = train_out[start:last*100]
    start+=batchSize
    last+=i
    # perm = np.random.permutation(batchSize)
    # batch_xs = [batch_xs[i] for i in perm]
    # batch_ys= [batch_ys[i] for i in perm]
    _, loss = sess.run([train_step, cross_entropy], feed_dict={x: batch_xs, y_: batch_ys})
    print(str(i) + 'th loss: ' + str(loss))


### Evaluation
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

# Check accuracy.
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: test_in, y_: test_out}))

######

import tensorflow as tf

#define a variable to hold normal random values
normal_rv = tf.Variable( tf.truncated_normal([2,3],stddev = 0.1))

#initialize the variable
init_op = tf.global_variables_initializer()

#run the graph
with tf.Session() as sess:
    sess.run(init_op) #execute init_op
    #print the random values that we sample
    print (sess.run(weightMatrix[0]))



import main.loadfile as lf
import numpy as np
import tensorflow as tf
import main.visualtools as vt
import main.setvalues as set
import main.networkmodels as net
# classification.
training = 'on';testing = 'on';fineTrainEpoch = 10;fineLearningRate = 0.01;batchSize = 100;normalize = 'off';hiddenLayers = [50,50];problem = 'classification' ;hiddenFunction= 'sigmoid'
costFunction = 'adam' ;plotOption = 'off';preTrainEpoch = 10;preLearningRate = 0.1;error_epoch = 5;inputData, targetData, test_in, test_out = lf.readmnist();RBM_values = set.setParam(inputData=inputData,targetData=targetData,hiddenUnits=hiddenLayers)
# Setting hidden layers: weightMatrix and biasMatrix
rbm_weightMatrix = RBM_values.genWeight()
rbm_biasMatrix = RBM_values.genBias()
rbm_input_x, rbm_input_y = RBM_values.genSymbol()
rbm = net.RBMmodel(inputSymbol=rbm_input_x,
                   preTrainEpoch=preTrainEpoch,
                   preLearningRate=preLearningRate,
                   weightMatrix=rbm_weightMatrix,
                   biasMatrix=rbm_biasMatrix,
                   batchSize=batchSize)
rbm.genRBM()
rbm.trainRBM(inputData)


rbm_vars = rbm.getVariables()


# LSTM test.
import _pickle as pickle
import main.loadfile as lf
import numpy as np
import tensorflow as tf
import main.visualtools as vt
import main.setvalues as set
import main.networkmodels as net
# classification.
training = 'on'
testing = 'on'
fineTrainEpoch = 10
fineLearningRate = 0.001
learningRateDecay = 'off' # on, off
batchSize = 100
normalize = 'off'
hiddenLayers = [100]
hiddenNumber = 2
problem = 'classification' # classification, regression
hiddenFunction= 'sigmoid'
costFunction = 'adam' # gradient, adam
plotOption = 'off'
preTrainEpoch = 10 # rbm epoch.
preLearningRate = 0.01 # rbm learning rate.
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import rnn

# Import MNIST data
# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Parameters
learning_rate = 0.001
training_iters = 100000
batchSize = 100
display_step = 10

# Network Parameters
inputDim = 39 # mfcc features
timeStep = 17 # timesteps
n_hidden = batchSize # hidden layer num of features
n_classes = 14 #

# inputData, targetData, test_in, test_out = lf.readmnist()
# mfcc: regression
# inputData, targetData, test_in, test_out = lf.readartmfcc()
# art and mfcc data(from jaegu)
with open("train_data/new_acoustics_vowel.pckl", "rb") as f:
    acoustics = pickle.load(f)
with open("train_data/new_articulation_vowel.pckl", "rb") as f:
    articulations = pickle.load(f)


print('Setting default parameters...')
# Setting default values by using SetValues.
lstm_values = set.setParam(inputData=acoustics,
                          targetData=articulations,
                          hiddenUnits=hiddenLayers
                          )

# Setting hidden layers: weightMatrix and biasMatrix
# lstm_weightMatrix = lstm_values.genWeight(option='random_normal')
# lstm_biasMatrix = lstm_values.genBias(option='random_normal')
# input_x = tf.placeholder("float", [None, inputDim, timeStep])
# input_y = tf.placeholder("float", [None, n_classes])
input_x = tf.placeholder(tf.float32,[batchSize,39])



# def RNN(inputs, weights, biases):
def genRNN(input_x,weight,bias):

    input = tf.transpose(input_x,[1,0,2])
    input = tf.reshape(input,[-1,inputDim])
    split_input = tf.split(input,timeStep,0)

    lstm_cell = rnn.BasicLSTMCell(hiddenLayers)
    rnn_cell = rnn.MultiRNNCell([lstm_cell] * hiddenNumber)
    initial_state = rnn_cell.zero_state(batchSize,tf.float32)

    output,state = tf.nn.dynamic_rnn(lstm_cell, split_input, initial_state=initial_state)


tmp = acoustics[0:100]
tmp = tf.transpose(tmp, [1, 0, 2])
# Reshaping to (n_steps*batch_size, n_input)
tmp = tf.reshape(tmp, [-1, inputDim])
# Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
split_inputs = tf.split(tmp, stepNumber, 0)

lstm_cell = rnn.BasicLSTMCell(hiddenLayers)

rnn_cell = rnn.MultiRNNCell([lstm_cell] * hiddenNumber)

state = lstm_cell.zero_state(batchSize,tf.float32)



# output, layers = tf.nn.dynamic_rnn(lstm_cell, split_inputs, [100], dtype=tf.float32)

# outputs, states = rnn.static_rnn(lstm_cell, input_x, dtype=tf.float32)

# Define a lstm cell with tensorflow
lstm_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)

# Get lstm cell output
outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

# Linear activation, using rnn inner loop last output
# return tf.matmul(outputs[-1], weights['out']) + biases['out']
tf.matmul(outputs[-1], weights['out']) + biases['out']

# pred = RNN(x, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.square(pred - y))
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 1
    # Keep training until reach max iterations
    while step * batchSize < training_iters:
        for start, last in zip(range(0, acoustics.shape[0], batchSize),
                               range(batchSize, acoustics.shape[0], batchSize)):
        batch_x = acoustics[start:last]
        batch_y = articulation[start:last]
        # Reshape data to get 28 seq of 28 elements
        batch_x = batch_x.reshape((batchSize, n_steps, n_input))
        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
        if step % display_step == 0:
            # Calculate batch accuracy
            acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
            # Calculate batch loss
            loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
            print("Iter " + str(step*batchSize) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
        step += 1
    print("Optimization Finished!")

    # Calculate accuracy for 128 mnist test images
    test_len = 128
    test_data = mnist.test.images[:test_len].reshape((-1, n_steps, n_input))
    test_label = mnist.test.labels[:test_len]
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={x: test_data, y: test_label}))



# #####
# max_Cycle = 100
# pattern_Count = 24023
#
# acoustics_Pattern = np.zeros((max_Cycle, pattern_Count, 39)).astype("float32")
# for index in range(len(acou)):
#     pattern = acou[index][:, 0:max_Cycle] # For test
#     # pattern = acou[index];
#     acoustics_Pattern[0:pattern.shape[1], index, :] = np.transpose(pattern)
#
# import matplotlib.pyplot as plt
# now_val = 0
# six_val = 0
# seven_val = 0
# record = np.array([0])
# for value in acoustics:
#     record = np.vstack((record, value.shape[1]))
#     if value.shape[1] == 16:
#         six_val += 1
#     elif value.shape[1] == 17:
#         seven_val += 1
#     if value.shape[1] > now_val:
#         now_val = value.shape[1]
#         best_val = now_val
# new_record = np.squeeze(record)
# plt.hist(new_record,30,range=[0,100])
# plt.hist(new_record,30,range=[0,50])
# plt.xticks(range(0,36,1))
# plt.xlabel("The length of time steps for each data.")
# plt.ylabel("The number of datasets")




# box = np.zeros([acoustics[0].shape[0],acoustics[0].shape[1]])
# for i in acoustics:
#     box = np.vstack((box,i))


# # data transformation
# # LSTM test.
# import _pickle as pickle
# import numpy as np
#
# with open("train_data/acoustics_vowel.pckl", "rb") as f:
#     acoustics = pickle.load(f)
# with open("train_data/articulation_vowel.pckl", "rb") as f:
#     articulation = pickle.load(f)
#
# new_acoustics = []
# new_articulation = []
#
# for aco_list,art_list in zip(acoustics,articulation):
#     step = aco_list.shape[1]
#     aco_item = aco_list.transpose()
#     art_item = art_list.transpose()
#     if step < 7 or step > 17:
#         pass
#         # new_acoustics.append(aco_item)
#         # new_articulation.append(art_item)
#     else:
#         if step != 17:
#             for turn in range(17-step):
#                 aco_item = np.vstack((aco_item,aco_item[-1]))
#                 art_item = np.vstack((art_item,art_item[-1]))
#         new_acoustics.append(aco_item)
#         new_articulation.append(art_item)
#
# pickle.dump(new_acoustics,open('new_acoustics.pckl','wb'))
# pickle.dump(new_articulation,open('new_articulation.pckl','wb'))


from __future__ import print_function
import _pickle as pickle
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn


# Import MNIST data
# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

with open("train_data/new_acoustics_vowel.pckl", "rb") as f:
    acoustics = pickle.load(f)
with open("train_data/new_articulations_vowel.pckl", "rb") as f:
    articulations = pickle.load(f)


# To classify images using a recurrent neural network, we consider every image
# row as a sequence of pixels. Because MNIST image shape is 28*28px, we will then
# handle 28 sequences of 28 steps for every sample.

def sigmoid(data):
    return 1 / (1 + np.exp(-0.25 * data))

# Parameters
learning_rate = 0.001
training_iters = 100000
batch_size = 100
display_step = 10

# Network Parameters
n_input = 39 # MNIST data input (img shape: 28*28)
n_steps = 17 # timesteps
n_hidden = 200 # hidden layer num of features
n_classes = 14 # MNIST total classes (0-9 digits)

# Define weights
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_classes]))
}

input_x = tf.placeholder(tf.float32,[None,n_steps,n_input])
input_y = tf.placeholder(tf.float32,[None,n_steps,n_classes])

# Define a lstm cell with tensorflow
lstm_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)

# Get lstm cell output
outputs, states = tf.nn.dynamic_rnn(lstm_cell, input_x, dtype=tf.float32)

pred_val = tf.matmul(outputs[-1], weights['out']) + biases['out']

# Define loss and optimizer
cost = tf.reduce_mean(tf.square(pred_val-input_y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
# correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
# accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 1
    # Keep training until reach max iterations
    start=0
    last=batch_size
    while step * batch_size < training_iters:
        batch_x = acoustics[start:last]
        batch_y = articulations[start:last]
        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={input_x: batch_x, input_y: batch_y})
        if step % display_step == 0:
            # Calculate batch accuracy
            # acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
            # Calculate batch loss
            loss = sess.run(cost, feed_dict={input_x: batch_x, input_y: batch_y})
            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) )
        step += 1
    print("Optimization Finished!")

    # Calculate accuracy for 128 mnist test images
    test_len = 128
    test_input = acoustics[0:1000]
    test_output = articulations[0:1000]
    print("Testing Accuracy:", \
        sess.run(cost, feed_dict={input_x: test_input, input_y: test_output}))




### Simple LSTM
import _pickle as pickle
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
with open("train_data/new_acoustics_vowel.pckl", "rb") as f:
    acoustics = pickle.load(f)
    # acoustics = dt.momentumSigmoid(acoustics,0.25)
    train_input = acoustics[0:18000]
    test_input = acoustics[18000:20001]
with open("train_data/new_articulations_vowel.pckl", "rb") as f:
    articulations = pickle.load(f)
    # articulations = dt.momentumSigmoid(articulations,0.1)
    train_output = articulations[0:18000]
    test_output = articulations[18000:20001]

# with open("train_data/acoustics.pckl", "rb") as f:
#     acoustics = pickle.load(f)
# with open("train_data/articulation.pckl", "rb") as f:
#     articulations = pickle.load(f)

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

lstm_net.trainLSTM(train_input,train_output)

lstm_net.testLSTM(test_input,test_output)

vars = lstm_net.getVariables()

lstm_net.closeLSTM()




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
    acoustics = dt.momentumSigmoid(acoustics,0.25)
with open("train_data/new_articulations_vowel.pckl", "rb") as f:
    articulations = pickle.load(f)


# Get lstm parameters.
lstm_values = set.LSTMParam(inputData=acoustics,
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

## lstm example with tensorflow
import numpy as np
from random import shuffle
import tensorflow as tf
import tensorflow.contrib.rnn as rnn

# generate input
train_input = ['{0:020b}'.format(i) for i in range(2 ** 20)]
shuffle(train_input)
train_input = [map(int, i) for i in train_input]
ti = []
for i in train_input:
    temp_list = []
    for j in i:
        temp_list.append([j])
    ti.append(np.array(temp_list))
train_input = ti

# generate output
train_output = []
for i in train_input:
    count = 0
    for j in i:
        if j[0] == 1:
            count += 1
    temp_list = ([0] * 21)
    temp_list[count] = 1
    train_output.append(temp_list)

# generate test data
NUM_EXAMPLES = 10000
test_input = train_input[NUM_EXAMPLES:]
test_output = train_output[NUM_EXAMPLES:]  # everything beyond 10,000

train_input = train_input[:NUM_EXAMPLES]
train_output = train_output[:NUM_EXAMPLES]  # till 10,000

data = tf.placeholder(tf.float32, [None, 20,1])
target = tf.placeholder(tf.float32, [None, 21])

num_hidden = 24
cell = rnn.BasicLSTMCell(num_hidden,state_is_tuple=True)

val, state = tf.nn.dynamic_rnn(cell, data, dtype=tf.float32)

val = tf.transpose(val, [1, 0, 2])
last = tf.gather(val, int(val.get_shape()[0]) - 1)

###### jk recipe
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Input/Ouput data
char_raw = 'hello_world_good_morning_see_you_hello_great'
char_list = list(set(char_raw))
char_to_idx = {c: i for i, c in enumerate(char_list)}
idx_to_char = {i: c for i, c in enumerate(char_list)}
char_data = [char_to_idx[c] for c in char_raw]
char_data_one_hot = tf.one_hot(char_data, depth=len(
    char_list), on_value=1., off_value=0., axis=1, dtype=tf.float32)
char_input = char_data_one_hot[:-1, :]  # 'hello_world_good_morning_see_you_hello_grea'
char_output = char_data_one_hot[1:, :]  # 'ello_world_good_morning_see_you_hello_great'
with tf.Session() as sess:
    char_input = char_input.eval()
    char_output = char_output.eval()


# Learning parameters
learning_rate = 0.001
max_iter = 200

# Network Parameters
n_input_dim = char_input.shape[1]
n_input_len = char_input.shape[0]
n_output_dim = char_output.shape[1]
n_output_len = char_output.shape[0]
n_hidden = 100

# TensorFlow graph
# (batch_size) x (time_step) x (input_dimension)
x = tf.placeholder(tf.float32, [10, None, n_input_dim])
# (batch_size) x (time_step) x (output_dimension)
y = tf.placeholder(tf.float32, [10, None, n_output_dim])

# Parameters
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, n_output_dim], seed=1))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_output_dim], seed=1))
}

# RNN-LSTM cell
def RNN(inputs, weights, biases):
    # Reshape to (time_step) x (batch_size) x (input_dimension)
    inputs = tf.transpose(inputs, [1, 0, 2])
    # Reshape to (time_step)*(batch_size) x (input_dimension)
    inputs = tf.reshape(inputs, [-1, n_input_dim])
    # Split to get a list of time_step tensors of shape (batch_size, input_dimension)
    # final 'inputs' is a list of n_input_len elements
    # (=number of frames)
    inputs = tf.split(value=inputs, num_or_size_splits=n_input_len, axis=0)

    lstm = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
    outputs, states = tf.nn.dynamic_rnn(lstm, x, dtype=tf.float32)
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

def softmax(x):
    rowmax = np.max(x, axis=1)
    x -= rowmax.reshape((x.shape[0] ,1)) # for numerical stability
    x = np.exp(x)
    sum_x = np.sum(x, axis=1).reshape((x.shape[0],1))
    return x / sum_x

pred = RNN(x, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

## learning.
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    train_x = char_input.reshape((1, char_input.shape[0], n_input_dim))
    train_y = char_output.reshape((1, char_output.shape[0], n_output_dim))
    for i in range(max_iter):
        _, loss, p = sess.run([optimizer, cost, pred],
                              feed_dict={x: train_x, y: train_y})
        if i is max_iter-1:
            pred_act = softmax(p)
        pred_out = np.argmax(p, axis=1)
        print('Epoch: {:>4}'.format(i + 1), '/', str(max_iter),
              'Cost: {:4f}'.format(loss), 'Predict:', ''.join([idx_to_char[i] for i in pred_out]))
        
'''
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import rnn

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

'''
To classify images using a recurrent neural network, we consider every image
row as a sequence of pixels. Because MNIST image shape is 28*28px, we will then
handle 28 sequences of 28 steps for every sample.
'''

# Parameters
learning_rate = 0.001
training_iters = 100000
batch_size = 128
display_step = 10

# Network Parameters
n_input = 28 # MNIST data input (img shape: 28*28)
n_steps = 28 # timesteps
n_hidden = 100 # hidden layer num of features
n_classes = 10 # MNIST total classes (0-9 digits)

# tf Graph input
x = tf.placeholder("float", [None, n_steps, n_input])
y = tf.placeholder("float", [None, n_classes])

# Define weights
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_classes]))
}


def RNN(x, weights, biases):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

    # Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    x = tf.unstack(x, n_steps, 1)

    # Define a lstm cell with tensorflow
    lstm_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)

    # Get lstm cell output
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

pred = RNN(x, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 1
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        # Reshape data to get 28 seq of 28 elements
        batch_x = batch_x.reshape((batch_size, n_steps, n_input))
        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
        if step % display_step == 0:
            # Calculate batch accuracy
            acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
            # Calculate batch loss
            loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
        step += 1
    print("Optimization Finished!")

    # Calculate accuracy for 128 mnist test images
    test_len = 128
    test_data = mnist.test.images[:test_len].reshape((-1, n_steps, n_input))
    test_label = mnist.test.labels[:test_len]
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={x: test_data, y: test_label}))

