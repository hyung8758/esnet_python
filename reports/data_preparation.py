'''
Data preparation for ANN & LSTM performance comparison.

                                                                    Written by Hyungwon Yang
                                                                                2016. 04. 17
                                                                                    EMCS Lab
'''

# Data preparation for ANN & LSTM performance comparison.

import re
import numpy as np



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



### word level
# This is first method. combine 3-gram. (1*5000) * 3 = 1 * 15000



# This is second method. put 3-gram into 1 vocabulary.
# ANN
ann_train_inputs = np.zeros((train_data_size,vocab_size))
ann_train_outputs = np.zeros((train_data_size,vocab_size))
box = 0
for idx in train_ngram:
    ann_train_inputs[box][idx[0:3]] = 1
    ann_train_outputs[box][idx[-1]] = 1
    box += 1
# test_inputs
ann_test_inputs = np.zeros((test_data_size,vocab_size))
ann_test_outputs = np.zeros((test_data_size,vocab_size))
box = 0
for idx in test_ngram:
    ann_test_inputs[box][idx[0:3]] = 1
    ann_train_outputs[box][idx[-1]] = 1
    box += 1


## LSTM
timeStep = 3

# dictionary list.
# Be cautious. This takes a lot of time to generate data.
word_box = np.identity(len(lookup_words),dtype=int)
input_box = np.zeros((timeStep,vocab_size))
lstm_train_inputs = np.empty((1,timeStep,vocab_size))
lstm_train_outputs = np.empty((1,timeStep,vocab_size))
con=0
for idx in train_ngram:
    input_box = np.zeros((timeStep,vocab_size))
    output_box = np.zeros((timeStep, vocab_size))
    for input in list(range(timeStep)):
        input_box[input][idx[input]] = 1
    for output in list(range(timeStep)):
        output_box[output][idx[output+1]] = 1
    lstm_train_inputs = np.append(lstm_train_inputs,[input_box],axis=0)
    lstm_train_outputs = np.append(lstm_train_outputs, [output_box], axis=0)
    con+=1
    if con % 500 == 0:
        print ('{} / {} is completed'.format(con,train_data_size))



########################################################################################################################
### char level
# data preprocessing.
# import data.
with open('train_data/pg8800_train_chars','r') as train_n:
    train_char = train_n.readlines()
    tmp_train = train_char[0].split(' ')
    train_split_char = tmp_train[0:170000]

with open('train_data/pg8800_test_chars','r') as test_n:
    test_char = test_n.readlines()
    tmp_test = test_char[0].split(' ')
    test_split_char = tmp_train[0:33000]

with open('train_data/pg8800_char_list','r') as look_w:
    lookup = look_w.readlines()
    lookup_chars = []
    for string in lookup:
        lookup_chars.append((re.sub('\n','',string)))


vocab_size = len(lookup_chars)
train_data_size = len(train_split_char)
test_data_size = len(test_split_char)

# data digitizing.
## ANN
# train data
ann_train_input_char = np.zeros((train_data_size-1,vocab_size))
ann_train_output_char = np.zeros((train_data_size-1,vocab_size))
for dat in list(range(train_data_size-1)):
    input_sym = train_split_char[dat]
    output_sym = train_split_char[dat+1]
    input_idx = lookup_chars.index(input_sym)
    output_idx = lookup_chars.index(output_sym)
    ann_train_input_char[dat][input_idx] = 1
    ann_train_output_char[dat][output_idx] = 1


# test data
ann_test_input_char = np.zeros((test_data_size - 1, vocab_size))
ann_test_output_char = np.zeros((test_data_size - 1, vocab_size))
for dat in list(range(test_data_size - 1)):
    input_sym = test_split_char[dat]
    output_sym = test_split_char[dat + 1]
    input_idx = lookup_chars.index(input_sym)
    output_idx = lookup_chars.index(output_sym)
    ann_test_input_char[dat][input_idx] = 1
    ann_test_output_char[dat][output_idx] = 1
np.savez('train_data/pg8800_ann_char_data',train_input=ann_train_input_char,
         train_output=ann_train_output_char,test_input=ann_test_input_char,test_output=ann_test_output_char)

## LSTM
lstm_train_input_char = np.zeros((int(train_data_size/timeStep),timeStep,vocab_size))
lstm_train_output_char = np.zeros((int(train_data_size/timeStep),timeStep,vocab_size))
his = 0
for dat in list(range(int(train_data_size/timeStep)-1)):
    for times in list(range(timeStep)):
        input_sym = train_split_char[his]
        output_sym = train_split_char[his+1]
        input_idx = lookup_chars.index(input_sym)
        output_idx = lookup_chars.index(output_sym)
        lstm_train_input_char[dat][times][input_idx] = 1
        lstm_train_output_char[dat][times][output_idx] = 1
        his += 1

# test data
lstm_test_input_char = np.zeros((int(test_data_size/timeStep),timeStep,vocab_size))
lstm_test_output_char = np.zeros((int(test_data_size/timeStep),timeStep,vocab_size))
his = 0
for dat in list(range(int(test_data_size/timeStep)-1)):
    for times in list(range(timeStep)):
        input_sym = test_split_char[his]
        output_sym = test_split_char[his+1]
        input_idx = lookup_chars.index(input_sym)
        output_idx = lookup_chars.index(output_sym)
        lstm_test_input_char[dat][times][input_idx] = 1
        lstm_test_output_char[dat][times][output_idx] = 1
        his += 1
np.savez('train_data/pg8800_lstm_char_data',train_input=lstm_train_input_char,
         train_output=lstm_train_output_char,test_input=lstm_test_input_char,test_output=lstm_test_output_char)
