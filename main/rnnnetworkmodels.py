'''
Machine Learning Network for RNN, LSTM, and GRU(will be supported soon).

                                                                    Written by Hyungwon Yang
                                                                                2016. 02. 07
                                                                                    EMCS Lab
'''
import numpy as np
import tensorflow as tf
import tensorflow.contrib.rnn as rnn

### RNN
'''
Simple means it has only one hidden layer.
This "simpleRNNmodel" will be updated to "RNNmodel" when it is fixed to support multiple hidden layers.
'''
class simpleRNNModel(object):

    def __init__(self,inputSymbol,outputSymbol,rnnCell,problem,trainEpoch,learningRate,learningRateDecay,timeStep,batchSize,
                 validationCheck,weightMatrix,biasMatrix):
        self.input_x = inputSymbol
        self.input_y = outputSymbol
        self.rnnCell = rnnCell
        self.problem = problem
        self.trainEpoch = trainEpoch
        self.lr = learningRate
        self.timeStep = timeStep
        self.batchSize = batchSize
        self.validationCheck = validationCheck
        self.weightMatrix = weightMatrix
        self.biasMatrix = biasMatrix
        # Setting for LearningRateDecay option.
        if learningRateDecay == 'on':
            self.lr = tf.train.exponential_decay(learningRate, self.trainEpoch * self.batchSize,
                                                         self.batchSize, 0.96, staircase=True)
        elif learningRateDecay == 'off':
            self.lr = learningRate if learningRate is not None else 0.01
        else:
            print('learningRateDecay option is not properly set. It will be off as a default.')
            self.lr = learningRate if learningRate is not None else 0.01


        # Check data type
        if inputSymbol.dtype == 'float32':
            self.dtype = tf.float32
        elif inputSymbol.dtype == 'float64':
            self.dtype = tf.float64
        else:
            raise ValueError('Input data type should be float for input and weight multiplication.')
        if self.rnnCell == 'rnn' or self.rnnCell == 'lstm' or self.rnnCell == 'gru':
            print('RNN cell type is',self.rnnCell)
        else: raise ValueError(self.rnnCell+' is not a correct cell type. Please provide rnn or lstm.')


    def genRNN(self):

        if self.rnnCell == 'rnn':
            rnn_cell = rnn.BasicRNNCell(self.weightMatrix.get_shape()[0].value)
        elif self.rnnCell == 'lstm':
            rnn_cell = rnn.BasicLSTMCell(self.weightMatrix.get_shape()[0].value, forget_bias=1.0)
        elif self.rnnCell == 'gru':
            rnn_cell = rnn.GRUCell(self.weightMatrix.get_shape()[0].value)

        # Get rnn cell output
        outputs, states = tf.nn.dynamic_rnn(rnn_cell, self.input_x, dtype=self.dtype)
        # batch_size * timeStep
        outputs = tf.reshape(outputs, [-1, outputs.get_shape()[2].value])
        self.pred_val = tf.matmul(outputs, self.weightMatrix) + self.biasMatrix

        # Define loss and optimizer
        if self.problem is 'classification':
            self.last_out = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.pred_val, labels=self.input_y))
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.last_out)
            self.correct_prediction = tf.equal(tf.argmax(self.pred_val, 1), tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, self.dtype))

        elif self.problem is 'regression':

            self.last_out = tf.reduce_mean(tf.square(self.pred_val - self.input_y))
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.last_out)


    def trainRNN(self,train_inputs,train_targets):

        # Initialize the variables.
        self.init = tf.global_variables_initializer()

        # Activate the graph.
        self.rnn_sess = tf.Session()
        self.rnn_sess.run(self.init)

        if self.validationCheck == 'on':
            train_num = train_inputs.shape[0]
            val_num = int(train_num * 0.2)
            val_start = train_num - val_num
            tmp_inputs = train_inputs
            tmp_targets = train_targets
            # distribute train and validation set.
            train_inputs = tmp_inputs[0:val_start]
            train_targets = tmp_targets[0:val_start]
            valid_inputs = tmp_inputs[val_start::]
            valid_targets = tmp_targets[val_start::]

        for epoch in range(self.trainEpoch):
            total_loss = []

            for start, last in zip(range(0, train_inputs.shape[0]+1, self.batchSize),
                                   range(self.batchSize, train_targets.shape[0]+1, self.batchSize)):
                batch_x = train_inputs[start:last]
                batch_y = train_targets[start:last]
                batch_y = np.reshape(batch_y,[-1, batch_y.shape[2]])
                _, loss = self.rnn_sess.run([self.optimizer,self.last_out],
                                        feed_dict={self.input_x: batch_x, self.input_y: batch_y})
                # Calculate mean loss.
                total_loss.append(loss)

            if self.validationCheck == 'on':
                if self.problem == 'classification':
                    if epoch == 0:
                        valid_targets = np.reshape(valid_targets, [-1, valid_targets.shape[2]])
                    result, self.y_hat = self.rnn_sess.run([self.accuracy, self.last_out],
                                                       feed_dict={self.input_x: valid_inputs, self.input_y: valid_targets})
                    print("Epoch: {:3d} / {:3d}, Cost : {:.6f}, Validation Accuracy: {:3.2f}%".format(epoch + 1,
                                                                                               self.trainEpoch,
                                                                                               np.mean(total_loss),
                                                                                               result * 100))
                elif self.problem == 'regression':
                    result, self.y_hat = self.rnn_sess.run([self.last_out, self.pred_val],
                                                       feed_dict={self.input_x: valid_inputs,self.input_y: valid_targets})
                    print("Epoch: {:3d} / {:3d}, Cost : {:.6f}, Validation Error: {:.4f}%".format(epoch + 1,
                                                                                            self.trainEpoch,
                                                                                            np.mean(total_loss),
                                                                                            result))
            else:
                print("Epoch: {:3d} / {:3d}, Cost : {:.6f}".format(epoch + 1,
                                                             self.trainEpoch,
                                                             np.mean(total_loss)))
        print("The model has been trained successfully.")

    def testRNN(self,test_inputs,test_targets):

        if self.problem is 'classification':
            test_targets = np.reshape(test_targets, [-1, test_targets.shape[2]])
            result, self.y_hat = self.rnn_sess.run([self.accuracy, self.last_out], feed_dict={self.input_x: test_inputs, self.input_y: test_targets})
            print("Tested with " + str(test_inputs.shape[0]) + " datasets.\n" + "Test Accuracy: " + "{:.2f}".format(result * 100) + " %")

        elif self.problem is 'regression':
            result, self.y_hat = self.rnn_sess.run([self.last_out,self.pred_val], feed_dict={self.input_x: test_inputs, self.input_y: test_targets})
            print("Tested with " + str(test_targets.shape[2])+ " datasets.\n" + "Test Error: " + "{:.4f}".format(result) + " %")


    def getVariables(self):
        var_dict = {}
        var_dict["weight"] = self.weightMatrix
        var_dict["bias"] = self.biasMatrix
        var_dict["y_hat"] = self.y_hat
        var_dict["optimizer"] = self.optimizer
        var_dict["cost"] = self.last_out
        print("Variable list as a dictionary format.\n>> weight, bias, y_hat, optimizer, cost\n")

        return var_dict

    def closeRNN(self):
        self.rnn_sess.close()
        print("Simple RNN training session is terminated.")

