"""
Manipulating data for them to be fit into training process.

                                                                               Hyungwon Yang
                                                                                2016. 02. 10
                                                                                   EMCS Labs
"""
import numpy as np


def normalize(data, type, axis=0):

    if type == 'sigmoid':
        tmp_data = 1 / (1 + np.exp(-data))
        return [tmp_data]
    elif type == 'tanh':
        tmp_data = np.tanh(data)
        return [tmp_data]
    elif type == 'zscore':
        mu = np.mean(data, axis, keepdims=True)
        std = np.std(data, axis, keepdims=True)
        out = (data - mu) / std
        return [out, mu, std]
    elif type == 'minmax':
        min_val = np.amin(data, axis=axis)
        max_val = np.amax(data, axis=axis)
        out = (data - min_val) * 2 / max_val - 1
        return [out, min_val, max_val]


def momentumSigmoid(data,momentum):
    return 1 / (1 + np.exp(-momentum * data))


# Make raw text file that can be trainable in ANN (# of example * # of feature.)
def text2rnnonehot(text, delimiter=' ', timestep=0):
    with open(text, 'r') as train_n:
        in_txt = train_n.readlines()
        tmp_txt = in_txt[0].split(delimiter)
        uniq_word = list(set(tmp_txt))

    vocab_size = len(uniq_word)
    data_size = len(tmp_txt)

    # Generate dataset.
    # ann
    if timestep == 0:
        input_txt = np.zeros((data_size, vocab_size))
        output_txt = np.zeros((data_size, vocab_size))
        for dat in range(data_size):
            if dat != (data_size - 1):
                input_sym = tmp_txt[dat]
                output_sym = tmp_txt[dat + 1]
                input_idx = uniq_word.index(input_sym)
                output_idx = uniq_word.index(output_sym)
                input_txt[dat][input_idx] = 1
                output_txt[dat][output_idx] = 1
            else:
                input_sym = tmp_txt[dat]
                input_idx = uniq_word.index(input_sym)
                input_txt[dat][input_idx] = 1

        data_shape = (data_size, vocab_size)

        print('Text file format is transformed successfully.')
        print("Dataset dimension: examples: {}, features: {}".format(data_shape[0], data_shape[1]))
        return input_txt, output_txt

    # rnn
    else:
        if data_size%timestep != 0:
            print("WARNNING: DATA CLIPPING")
            print("Input data cannot be clearly divided into timeStep:{}. Last {} examples will be discarded."
                  .format(timestep, data_size%timestep))
            print("Provide different number of timeStep to save input data.")
        his = 0
        input_txt = np.zeros((int(data_size / timestep), timestep, vocab_size))
        output_txt = np.zeros((int(data_size / timestep), timestep, vocab_size))
        for dat in list(range(int(data_size/timestep)-1)):
            for times in list(range(timestep)):
                input_sym = tmp_txt[his]
                output_sym = tmp_txt[his + 1]
                input_idx = uniq_word.index(input_sym)
                output_idx = uniq_word.index(output_sym)
                input_txt[dat][times][input_idx] = 1
                output_txt[dat][times][output_idx] = 1
                his += 1

        data_shape = input_txt.shape
        print('Text file format is transformed successfully.')
        print("Dataset dimension: examples: {}, timeStep: {}, features: {}"
              .format(data_shape[0], data_shape[1], data_shape[2]))
        return input_txt, output_txt