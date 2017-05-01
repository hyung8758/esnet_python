'''
Machine Learning Network for DNN and RBM.

                                                                    Written by Hyungwon Yang
                                                                                2016. 02. 07
                                                                                    EMCS Lab
'''
import numpy as np
import tensorflow as tf
import main.visualtools as vt

### DNN
class DNNmodel(object):

    def __init__(self,inputSymbol,outputSymbol,problem,fineTrainEpoch,fineLearningRate,learningRateDecay,batchSize,
                      hiddenFunction,costFunction,validationCheck,plotGraph,weightMatrix,biasMatrix):
        self.input_x = inputSymbol
        self.input_y = outputSymbol
        self.problem = problem
        self.fineTrainEpoch = fineTrainEpoch
        self.batchSize = batchSize
        self.hiddenFunction = hiddenFunction
        self.costFunction = costFunction
        self.validationCheck = validationCheck
        self.plotGraph = plotGraph
        self.weightMatrix = weightMatrix
        self.biasMatrix = biasMatrix
        # Setting for LearningRateDecay option.
        if learningRateDecay == 'on':
            self.finelr = tf.train.exponential_decay(fineLearningRate, self.fineTrainEpoch * self.batchSize,
                                                         self.batchSize, 0.96, staircase=True)
        elif learningRateDecay == 'off':
            self.finelr = fineLearningRate if fineLearningRate is not None else 0.01
        else:
            print('learningRateDecay option is not properly set. It will be off as a default.')
            self.finelr = fineLearningRate if fineLearningRate is not None else 0.01

    def genDNN(self):
        '''
        This feedforward network is not intended for batch_learning. Instead, it is on-lines based learning networks
        that take more learning time but update weights and biases frequently.
        '''

        # iteration
        hiddenNumber = len(self.weightMatrix)-1
        outputStorage = []
        outputActivation = []

        # First up
        outputStorage.append(tf.matmul(self.input_x,self.weightMatrix[0]) + self.biasMatrix[1])
        if self.hiddenFunction is 'sigmoid':
            outputActivation.append(tf.nn.sigmoid(outputStorage[0]))

            # Hidden Layer up
            for iter in range(1,hiddenNumber):

                # Hiddenlayer activation.
                outputStorage.append(tf.matmul(outputActivation[iter-1],self.weightMatrix[iter]) + self.biasMatrix[iter+1])
                outputActivation.append(tf.nn.sigmoid(outputStorage[iter]))

        elif self.hiddenFunction is 'tanh':
            outputActivation.append(tf.nn.tanh(outputStorage[0]))

            # Hidden layer up
            for iter in range(1,hiddenNumber):

                # Hiddenlayer activation.
                outputStorage.append(tf.matmul(outputActivation[iter-1],self.weightMatrix[iter]) + self.biasMatrix[iter+1])
                outputActivation.append(tf.nn.tanh(outputStorage[iter]))

        # Last up with problem selection.
        if self.problem is 'classification':
            self.last_out = tf.matmul(outputActivation[-1],self.weightMatrix[-1]) + self.biasMatrix[-1]
            self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.last_out, labels=self.input_y))
            correct_prediction = tf.equal(tf.argmax(self.last_out, 1), tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        elif self.problem is 'regression':
            self.last_out = tf.matmul(outputActivation[-1], self.weightMatrix[-1]) + self.biasMatrix[-1]
            self.cost = tf.reduce_mean(tf.square(self.last_out - self.input_y))

        # Select cost function
        if self.costFunction is 'gradient':
            self.optimizer = tf.train.GradientDescentOptimizer(self.finelr).minimize(self.cost)
        elif self.costFunction is 'adam':
            self.optimizer = tf.train.AdamOptimizer(self.finelr).minimize(self.cost)


    # Train DNN network.
    def trainDNN(self,train_inputs,train_targets):

        # Initialize the variables.
        self.init = tf.global_variables_initializer()

        # Activate the graph.
        self.dnn_sess = tf.Session()
        self.dnn_sess.run(self.init)

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

        if self.plotGraph == 'on':
            writer = tf.summary.FileWriter('./dnn_graphs', self.dnn_sess.graph)
            writer.close()

        for epoch in range(self.fineTrainEpoch):
            cost_mean = 0
            total_batch = int(train_inputs.shape[0]/self.batchSize)

            for start, last in zip(range(0, train_inputs.shape[0]+1, self.batchSize), range(self.batchSize, train_targets.shape[0]+1, self.batchSize)):
                batch_x = train_inputs[start:last]
                batch_y = train_targets[start:last]
                _, loss = self.dnn_sess.run([self.optimizer, self.cost], feed_dict={self.input_x: batch_x, self.input_y: batch_y})

                # Calculate mean loss.
                cost_mean += loss/total_batch

            if self.validationCheck == 'on':
                if self.problem == 'classification':
                    result, self.y_hat = self.dnn_sess.run([self.accuracy, self.last_out],
                                                       feed_dict={self.input_x: valid_inputs, self.input_y: valid_targets})
                    print("Epoch: {:3d} / {:3d}, Cost : {:.6f}, Validation Accuracy: {:3.2f}%".format(epoch + 1,
                                                                                               self.fineTrainEpoch,
                                                                                               np.mean(cost_mean),
                                                                                               result * 100))
                elif self.problem == 'regression':
                    result = self.dnn_sess.run(self.cost, feed_dict={self.input_x: valid_inputs,self.input_y: valid_targets})
                    print("Epoch: {:3d} / {:3d}, Cost : {:.6f}, Validation Error: {:.4f}%".format(epoch + 1,
                                                                                            self.fineTrainEpoch,
                                                                                            np.mean(cost_mean),
                                                                                            result))
            else:
                print("Epoch: {:3d} / {:3d}, Cost : {:.6f}".format(epoch + 1,
                                                             self.fineTrainEpoch,
                                                             np.mean(cost_mean)))

        print("The model has been trained successfully.")

        # Update weight and bias.
        self.updated_weightMatrix = self.dnn_sess.run(self.weightMatrix)
        self.updated_biasMatrix = self.dnn_sess.run(self.biasMatrix)

    # Test trained DNN network.
    def testDNN(self,test_inputs,test_targets):

        # Check accuracy.
        if self.problem is 'classification':
            result, self.y_hat = self.dnn_sess.run([self.accuracy,self.last_out], feed_dict={self.input_x: test_inputs, self.input_y: test_targets})
            print("Tested with ",str(test_inputs.shape[0])," datasets.\n" + "Test Accuracy: " + "{:.2f}".format(result*100) + " %")

        elif self.problem is 'regression':
            result = self.dnn_sess.run(self.cost, feed_dict={self.input_x: test_inputs, self.input_y: test_targets})
            print("Tested with " + str(test_inputs.shape[0]) + " datasets.\n" + "Test error: " + "{:.4f}".format(result))


    def getVariables(self):
        var_dict = {}
        var_dict["weight"] = self.updated_weightMatrix
        var_dict["bias"] = self.updated_biasMatrix
        var_dict["y_hat"] = self.y_hat
        var_dict["optimizer"] = self.optimizer
        var_dict["cost"] = self.cost
        print("Variable list as a dictionary format.\n>> weight, bias, y_hat, optimizer, cost\n")

        return var_dict

    def closeDNN(self):
        self.dnn_sess.close()
        tf.reset_default_graph()
        print("DNN training session is terminated.")


### RBM
class RBMmodel(object):

    def __init__(self,inputSymbol,preTrainEpoch,preLearningRate,batchSize,weightMatrix,biasMatrix):
        self.input_x = inputSymbol
        self.preTrainEpoch = preTrainEpoch
        self.prelr = preLearningRate
        self.batchSize = batchSize
        self.weightMatrix = weightMatrix
        self.biasMatrix = biasMatrix
        self.hiddenNumber = len(weightMatrix)-1
        self.next_inputs = []

        # generate placeholders.
        if self.weightMatrix[0].dtype == 'float32':
            self.vhMatrix = tf.placeholder('float32',[None,None])
            self.hBiasMatrix = tf.placeholder('float32',[None,None])
            self.vBiasMatrix = tf.placeholder('float32',[None,None])
        elif self.biasMatrix[0].dtype == 'float64':
            self.vhMatrix = tf.placeholder('float64',[None,None])
            self.hBiasMatrix = tf.placeholder('float64',[None,None])
            self.vBiasMatrix = tf.placeholder('float64',[None,None])

    def genRBM(self):
        # Generate RBM structure.
        # hidden0
        hidden0Array = tf.nn.sigmoid(tf.matmul(self.input_x,self.vhMatrix) + self.hBiasMatrix)

        # visual1
        visual1Array = tf.nn.sigmoid(tf.matmul(hidden0Array,tf.transpose(self.vhMatrix)) + self.vBiasMatrix)

        # hidden1
        hidden1Array = tf.nn.sigmoid(tf.matmul(visual1Array,self.vhMatrix) + self.hBiasMatrix)

        # update weights and biases
        self.vhMatrix_cost = self.vhMatrix + self.prelr * (tf.matmul(tf.transpose(self.input_x),hidden0Array) - tf.matmul(tf.transpose(visual1Array),hidden1Array))
        self.vBiasMatrix_cost = self.vBiasMatrix + self.prelr * tf.reduce_mean(self.input_x - visual1Array,0)
        self.hBiasMatrix_cost = self.hBiasMatrix + self.prelr * tf.reduce_mean(hidden0Array - hidden1Array,0)
        self.next_inputs = hidden0Array

        # Error calculation.
        error = self.input_x - visual1Array
        self.total_error = tf.reduce_mean(tf.square(error))


    # Train RBM network.
    def trainRBM(self,train_inputs):

        # Initialize the variables.
        self.init = tf.global_variables_initializer()

        # Activate the graph.
        self.rbm_sess = tf.Session()
        self.rbm_sess.run(self.init)

        for hid in range(self.hiddenNumber):
            # Distribute variables.
            if hid == 0:
                tmp_weight = vt.printvar(self.weightMatrix[hid],echo='off')
                tmp_vbias = vt.printvar(self.biasMatrix[hid],echo='off')
                tmp_hbias = vt.printvar(self.biasMatrix[hid+1],echo='off')
                all_inputs = np.zeros((self.batchSize,tmp_weight.shape[1]))
            else:
                tmp_weight = vt.printvar(self.weightMatrix[hid],echo='off')
                tmp_vbias = self.biasMatrix[hid]
                tmp_hbias = vt.printvar(self.biasMatrix[hid+1],echo='off')
                all_inputs = np.zeros((self.batchSize,tmp_weight.shape[1]))

            for epoch in range(self.preTrainEpoch):
                saved_error = np.array(0,'float32')
                if epoch+1 != self.preTrainEpoch:
                    for start, last in zip(range(0, train_inputs.shape[0]+1, self.batchSize), range(self.batchSize, train_inputs.shape[0]+1, self.batchSize)):
                        self.new_W, self.new_hB, self.new_vB, new_inputs, err=self.rbm_sess.run([self.vhMatrix_cost, self.hBiasMatrix_cost, self.vBiasMatrix_cost,self.next_inputs,self.total_error],
                                                            feed_dict={self.input_x: train_inputs[start:last], self.vhMatrix: tmp_weight,
                                                                       self.vBiasMatrix: tmp_vbias, self.hBiasMatrix: tmp_hbias})
                        # Update weight and Bias.
                        tmp_weight = self.new_W
                        tmp_vbias = self.new_vB
                        tmp_hbias = self.new_hB
                        # Stacking error.
                        saved_error = np.vstack((saved_error,err))

                # At the last epoch.
                elif epoch+1 == self.preTrainEpoch:
                    for start, last in zip(range(0, train_inputs.shape[0]+1, self.batchSize), range(self.batchSize, train_inputs.shape[0]+1, self.batchSize)):
                        self.new_W, self.new_hB, self.new_vB, new_inputs = self.rbm_sess.run([self.vhMatrix_cost, self.hBiasMatrix_cost, self.vBiasMatrix_cost,self.next_inputs],
                                                            feed_dict={self.input_x: train_inputs[start:last], self.vhMatrix: tmp_weight,
                                                                       self.vBiasMatrix: tmp_vbias, self.hBiasMatrix: tmp_hbias})
                        # Update weight and Bias.
                        tmp_weight = self.new_W
                        tmp_vbias = self.new_vB
                        tmp_hbias = self.new_hB
                        # Collect New hidden result for next inputs.
                        all_inputs = np.vstack((all_inputs,new_inputs))
                        # Stacking error.
                        saved_error = np.vstack((saved_error,err))

                    # Get all the newly generated inputs for next hidden layer's inputs.
                print('{:3d}/{:3d} Hidden Layer, {:3d}/{:3d} epoch, MSE: {}'.format(hid + 1, self.hiddenNumber, epoch + 1, self.preTrainEpoch,str(np.mean(saved_error))))
                # Change input value.
                if epoch+1 == self.preTrainEpoch:
                    cut_zeros_inputs = all_inputs[self.batchSize::,:]
                    train_inputs = cut_zeros_inputs
                    self.weightMatrix[hid] = self.new_W
                    self.biasMatrix[hid] = self.new_vB
                    self.biasMatrix[hid+1] = self.new_hB

        print("The model has been trained successfully.")


    def getVariables(self):
        var_dict = {}
        var_dict["weight"] = self.weightMatrix
        var_dict["bias"] = self.biasMatrix
        print("Variable list as a dictionary format.\n>> weight, bias\n")

        return var_dict

    def closeRBM(self):
        self.rbm_sess.close()
        tf.reset_default_graph()
        print("RBM training session is terminated.")






