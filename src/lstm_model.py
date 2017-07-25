'''
LSTM network for truncated BPTT training. 
                                                            Hyungwon Yang
'''


import numpy as np
import tensorflow as tf
import src.loadfile as lf

# Import data
train_input, train_output, test_input, test_output = lf.readpg8800rnnchar()

# Parameters
input_dim = train_output.shape[2]
output_dim = train_input.shape[2]
hiddenUnits = 200
timeStep = 20
btpp_step = 5


# weight and bias
with tf.variable_scope('lstm_cell'):
    # weights for input to hidden.
    input_w = tf.get_variable("input_w",[input_dim,hiddenUnits],initializer=tf.random_normal_initializer())
    output_w = tf.get_variable('output_w',[input_dim,hiddenUnits],initializer=tf.random_normal_initializer())
    forget_w = tf.get_variable('forget_w',[input_dim,hiddenUnits],initializer=tf.random_normal_initializer())
    tmp_cell_w = tf.get_variable('tmp_cell_w', [input_dim, hiddenUnits], initializer=tf.random_normal_initializer())
    final_w = tf.get_variable('final_w',[hiddenUnits,output_dim], initializer=tf.random_normal_initializer())
    # weights for hidden to hidden.
    i_hidden_w = tf.get_variable("i_hidden_w", [hiddenUnits, hiddenUnits], initializer=tf.random_normal_initializer())
    o_hidden_w = tf.get_variable('o_hidden_w', [hiddenUnits, hiddenUnits], initializer=tf.random_normal_initializer())
    f_hidden_w = tf.get_variable('f_hidden_w', [hiddenUnits, hiddenUnits], initializer=tf.random_normal_initializer())
    tc_hidden_w = tf.get_variable('tc_hidden_w', [hiddenUnits, hiddenUnits], initializer=tf.random_normal_initializer())
    # bias
    input_b = tf.get_variable("input_b", [hiddenUnits], initializer=tf.constant_initializer())
    output_b = tf.get_variable("output_b", [hiddenUnits], initializer=tf.constant_initializer())
    forget_b = tf.get_variable("forget_b", [hiddenUnits], initializer=tf.constant_initializer())
    tmp_cell_b = tf.get_variable("tmp_cell_b", [hiddenUnits], initializer=tf.constant_initializer())
    final_b = tf.get_variable("final_b", [output_dim], initializer=tf.constant_initializer())


# placeholder for input and output
input_x = tf.placeholder(train_input.dtype,[None, timeStep, input_dim])
input_y = tf.placeholder(train_input.dtype,[None, output_dim])

# Feed-forward functions.
def input_gate(train_input,i_hidden_w,input_w,prev_hid,input_b):
    z = tf.matmul(i_hidden_w,prev_hid) + tf.matmul(input_w,train_input) + input_b
    return tf.sigmoid(z)

def output_gate(train_input,o_hidden_w,output_w,prev_hid,output_b):
    z = tf.matmul(o_hidden_w,prev_hid) + tf.matmul(output_w,train_input) + output_b
    return tf.sigmoid(z)

def forget_gate(train_input,f_hidden_w,forget_w,prev_hid,forget_b):
    z = tf.matmul(f_hidden_w,prev_hid) + tf.matmul(forget_w,train_input) + forget_b
    return tf.sigmoid(z)

def tmp_cell_gate(train_input,tc_hidden_w,tmp_cell_w,prev_hid,tmp_cell_b):
    z = tf.matmul(tc_hidden_w,prev_hid) + tf.matmul(tmp_cell_w,train_input) + tmp_cell_b
    return tf.tanh(z)

def cell_gate(forget_val,cell_val,input_val,tmp_cell_val):
    return tf.multiply(forget_val,cell_val) + tf.multiply(input_val,tmp_cell_val)

def hidden_gate(output_val,tmp_cell_val):
    return tf.multiply(output_val,tf.tanh(tmp_cell_val))

def final_gate(hidden_val,final_w,final_b):
    return tf.matmul(hidden_val,final_w) + final_b


def build_graph():

    # RNN cell: Feedforward.
    hidden_state = []
    cell_state = []
    input_state = []
    output_state = []
    forget_state = []
    tmp_cell_state = []
    final_state = []
    pred = []
    loss = []

    new_hidden = np.zeros((hiddenUnits,hiddenUnits))
    new_cell = np.zeros(hiddenUnits)

    for ts in range(timeStep):
        # Input, output, forget gate.(state)
        input_state.append(input_gate(input_x[:,ts],i_hidden_w,input_w,new_hidden,input_b))
        output_state.append(output_gate(input_x[:,ts],o_hidden_w,output_w,new_hidden,output_b))
        forget_state.append(forget_gate(input_x[:,ts], f_hidden_w, forget_w, new_hidden, forget_b))
        # Two cell gates(state).
        tmp_cell_state.append(tmp_cell_gate(input_x[:,ts], tc_hidden_w, tmp_cell_w, new_hidden, tmp_cell_b))
        new_cell = cell_gate(input_x[:,ts],forget_state[ts],tmp_cell_state[ts],new_cell)
        cell_state.append(new_cell)
        # Hidden gate(state).
        new_hidden = hidden_gate(output_state[ts],tmp_cell_state[ts])
        hidden_state.append(new_hidden)

        # Final gate(state): predictions
        final_state.append(final_gate(new_hidden,final_w,final_b))
        pred.append(tf.nn.softmax(final_state[ts]))
        loss.append(input_y[:,ts] - pred[ts])

    total_loss = tf.reduce_mean(loss)

    # Error calculation and Truncated BPTT.
    



