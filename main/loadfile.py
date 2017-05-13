# -*- coding: utf-8 -*-
'''

loadfile.py is written for importing data conveniently.

- Curvefitting (function name / data name)
	1. readbody / bodyData.mat
	2. readbuilding / buildingData.mat
	3. readartmfcc / art_mfcc.mat
	4. readartandacou / new_acoustics.pckl and new_articulation.pckl

- Classification
	1. readmnist / MNIST (4 datasets)
	2. readcancer / cancerData.mat
	3. readpg8800rnnchar / 

																	   		   Hyungwon Yang
																				2016. 02. 10
																					EMCS Lab
'''

import scipy.io as sio
import numpy as np
import requests
import pickle
import os

data_path = './train_data'

def download_file_from_google_drive(id, destination):
    URL = "https://drive.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

################################################## CurveFitting Data ##################################################

# Import bodyfatData
def readbody():

	file_check = os.path.exists(data_path+'/bodyData.mat')
	if file_check is False:
		print('bodyData.mat is not present in the train_data directory.')

		print("Downloading bodyData.mat...")
		datafile = "0B9lwe_GFwe2oV2lhbS1PNFdKbnM"
		savefile = "train_data/bodyData.mat"
		download_file_from_google_drive(datafile, savefile)

		print("Dataset Downloaded successfully. Check train_data folder.")

	source_path = os.path.join(data_path,'bodyData.mat')
	data = sio.loadmat(source_path)

	bodyfatInputs = data['bodyfatInputs'].transpose()
	bodyfatTargets = data['bodyfatTargets'].transpose()

	train_output = np.float32(bodyfatTargets[0:200])
	train_input = np.float32(bodyfatInputs[0:200])
	test_output = np.float32(bodyfatTargets[200:253])
	test_input = np.float32(bodyfatInputs[200:253])

	print('Data Information')
	print('Train data: 200 examples, Test data: 52 examples\nInputs: 13 features, Outputs: 1 feature\n')
	return train_input, train_output, test_input, test_output

# Import buildingData
def readbuilding():

	file_check = os.path.exists(data_path+'/buildingData.mat')
	if file_check is False:
		print('buildingData.mat is not present in the train_data directory.')
		print("Downloading buildingData.mat...")
		datafile = "0B9lwe_GFwe2oMmxPalZSY1pEMkU"
		savefile = "train_data/buildingData.mat"
		download_file_from_google_drive(datafile, savefile)

		print("Dataset Downloaded successfully. Check train_data folder.")

	source_path = os.path.join(data_path,'buildingData.mat')
	data = sio.loadmat(source_path)
	buildingInputs = data['buildingInputs'].transpose()
	buildingTargets = data['buildingTargets'].transpose()

	train_output = np.float32(buildingTargets[0:3400])
	train_input = np.float32(buildingInputs[0:3400])
	test_output = np.float32(buildingTargets[3400:4209])
	test_input = np.float32(buildingTargets[3400:4209])

	print('Data Information')
	print('Train data: 3,400 examples, Test data: 808 examples\nInputs: 14 features, Outputs: 3 features\n')
	return train_input, train_output, test_input, test_output

# Import art_mfcc Data
def readartmfcc():

	file_check = os.path.exists(data_path+'/art_mfcc.mat')
	if file_check is False:
		print('art_mfcc.mat is not present in the train_data directory.')
		print("Downloading art_mfcc.mat...")
		datafile = "0B9lwe_GFwe2oOXFqVzBlSWFYdDA"
		savefile = "train_data/art_mfcc.mat"
		download_file_from_google_drive(datafile, savefile)

		print("Dataset Downloaded successfully. Check train_data folder.")

	source_path = os.path.join(data_path,'art_mfcc.mat')
	data = sio.loadmat(source_path)
	art_V = data['art_V'].transpose()
	mfcc_V = data['mfcc_V'].transpose()

	train_output = np.float32(art_V[0:12000])
	train_input = np.float32(mfcc_V[0:12000])
	test_output = np.float32(art_V[12000:16001])
	test_input = np.float32(mfcc_V[12000:16001])

	print('Data Information')
	print('Train data: 12,000 examples, Test data: 4,000 examples\nInputs: 39 features, Outputs: 16 features(x&y pairs, total 8 pallet locations)\n')
	return train_input, train_output, test_input, test_output


# Import articulations and acoustics data for rnn.
def readartandacou():

	file_check = os.path.exists(data_path+'/new_acoustics.pckl')
	if file_check is False:
		print('new_acoustics.pckl is not present in the train_data directory.')
		print("Downloading art_new_acoustics.pckl...")
		datafile = "0B9lwe_GFwe2oS2lTZG1oQ2ZCejQ"
		savefile = "train_data/new_acoustics.pckl"
		download_file_from_google_drive(datafile, savefile)

		print("Dataset Downloaded successfully. Check train_data folder.")

	file_check = os.path.exists(data_path+'/new_articulation.pckl')
	if file_check is False:
		print('new_articulation.pckl is not present in the train_data directory.')
		print("Downloading art_new_articulation.pckl...")
		datafile = "0B9lwe_GFwe2oMkFVdzE1QXhOdG8"
		savefile = "train_data/new_articulation.pckl"
		download_file_from_google_drive(datafile, savefile)

		print("Dataset Downloaded successfully. Check train_data folder.")

	with open("train_data/new_acoustics.pckl", "rb") as f:
		acoustics = pickle.load(f)
		train_input = np.reshape(acoustics[0:18000], [18000, 17, 39])
		test_input = np.reshape(acoustics[18000:20000], [2000, 17, 39])
	with open("train_data/new_articulation.pckl", "rb") as f:
		articulations = pickle.load(f)
		train_output = np.reshape(articulations[0:18000], [18000, 17, 14])
		test_output = np.reshape(articulations[18000:20000], [2000, 17, 14])

	print('Data Information')
	print('Train data: 18,000 examples and 39 features, Test data: 2,000 examples and 14 featrues.\nTime steps for both data: 17')
	return train_input, train_output, test_input, test_output



################################################# Classification Data #################################################

# Import MNIST
def one_hot(x,n):
	if type(x) == list:
		x = np.array(x)
	x = x.flatten()
	o_h = np.zeros((len(x),n))
	o_h[np.arange(len(x)),x] = 1
	return o_h

def readmnist(ntrain=60000,ntest=10000,onehot=True):

	file_check = os.path.exists(data_path+'/train-labels-idx1-ubyte')
	if file_check is False:
		print('MNIST is not present in the train_data directory.')
		print("Downloading MNIST...")

		print("Downloading train-labels-idx1-ubyte...")
		datafile = "0B9lwe_GFwe2oZFRFX21zZUJQV1U"
		savefile = "train_data/train-labels-idx1-ubyte"
		download_file_from_google_drive(datafile, savefile)

		print("Downloading train-images-idx3-ubyte...")
		datafile = "0B9lwe_GFwe2oWmZmYnA3bG8yWGc"
		savefile = "train_data/train-images-idx3-ubyte"
		download_file_from_google_drive(datafile, savefile)

		print("Downloading t10k-labels-idx1-ubyte...")
		datafile = "0B9lwe_GFwe2oTmhWenVSUS00eTA"
		savefile = "train_data/t10k-labels-idx1-ubyte"
		download_file_from_google_drive(datafile, savefile)

		print("Downloading t10k-images-idx3-ubyte...")
		datafile = "0B9lwe_GFwe2oWHFNT0NPdVBEcms"
		savefile = "train_data/t10k-images-idx3-ubyte"
		download_file_from_google_drive(datafile, savefile)

		print("Dataset Downloaded successfully. Check train_data folder.")

	fd = open(os.path.join(data_path,'train-images-idx3-ubyte'))
	loaded = np.fromfile(file=fd,dtype=np.uint8)
	trX = loaded[16:].reshape((60000,28*28)).astype(float)

	fd = open(os.path.join(data_path,'train-labels-idx1-ubyte'))
	loaded = np.fromfile(file=fd,dtype=np.uint8)
	trY = loaded[8:].reshape((60000))

	fd = open(os.path.join(data_path,'t10k-images-idx3-ubyte'))
	loaded = np.fromfile(file=fd,dtype=np.uint8)
	teX = loaded[16:].reshape((10000,28*28)).astype(float)

	fd = open(os.path.join(data_path,'t10k-labels-idx1-ubyte'))
	loaded = np.fromfile(file=fd,dtype=np.uint8)
	teY = loaded[8:].reshape((10000))

	trX = trX/255.
	teX = teX/255.

	train_input = np.float32(trX[:ntrain])
	trY = trY[:ntrain]

	test_input = np.float32(teX[:ntest])
	teY = teY[:ntest]

	if onehot:
		train_output = np.float32(one_hot(trY, 10))
		test_output = np.float32(one_hot(teY, 10))
	else:
		train_output = np.float32(np.asarray(trY))
		test_output = np.float32(np.asarray(teY))

	print('Data Information')
	print('Train data:60,000 examples, Test data: 10,000 examples\nInputs: 784 features, Outputs: 10 classification(0-9 digits)\n')
	return train_input, train_output, test_input, test_output

# Import cancerData
def readcancer():

	file_check = os.path.exists(data_path+'/cancerData.mat')
	if file_check is False:
		print('cancerData.mat is not present in the train_data directory.')
		print("Downloading cancerData...")
		datafile = "0B9lwe_GFwe2oNEtSMHFXSWk3Rkk"
		savefile = "train_data/cancerData.mat"
		download_file_from_google_drive(datafile, savefile)

		print("Dataset Downloaded successfully. Check train_data folder.")

	source_path = os.path.join(data_path,'cancerData.mat')
	data = sio.loadmat(source_path)
	inputs = data['cancerInputs']
	outputs = data['cancerTargets']

	inputs = np.float64(inputs)
	outputs = np.float64(outputs)

	train_input = np.float32(inputs[:,0:550].transpose())
	train_output = np.float32(outputs[:,0:550].transpose())
	test_input = np.float32(inputs[:,550:701].transpose())
	test_output = np.float32(outputs[:,550:701].transpose())

	print('Data Information')
	print('Train data: 550 examples, Test data: 150 examples\nInputs: 9 features, Outputs: 2 classification(cancer or not)\n')
	return train_input, train_output, test_input, test_output

# Import pg8800 character level data for ann training.
def readpg8800annchar():
	file_check = os.path.exists(data_path + '/pg8800_ann_char_data.npz')
	if file_check is False:
		print('pg8800_ann_char_data.npz is not present in the train_data directory.')
		print("Downloading pg8800_ann_char_data.npz...")
		datafile = "0B9lwe_GFwe2oeWJlMDJKdjg2RXM"
		savefile = "train_data/pg8800_ann_char_data.npz"
		download_file_from_google_drive(datafile, savefile)

		print("Dataset Downloaded successfully. Check train_data folder.")

	rnn_data = np.load('train_data/pg8800_ann_char_data.npz')
	train_input = rnn_data['train_input'][0:160000]
	train_output = rnn_data['train_output'][0:160000]
	test_input = rnn_data['test_input'][0:32000]
	test_output = rnn_data['test_output'][0:32000]

	print('Data Information')
	print('Train data: 160,000 examples and 38 features, Test data: 32,000 examples and 38 features.')
	return train_input, train_output, test_input, test_output

# Import pg8800 character level data for rnn training.
def readpg8800rnnchar():
	file_check = os.path.exists(data_path + '/pg8800_rnn_char_data.npz')
	if file_check is False:
		print('pg8800_rnn_char_data.npz is not present in the train_data directory.')
		print("Downloading pg8800_rnn_char_data.npz...")
		datafile = "0B9lwe_GFwe2oYlpfbEdMVllOSmc"
		savefile = "train_data/pg8800_rnn_char_data.npz"
		download_file_from_google_drive(datafile, savefile)

		print("Dataset Downloaded successfully. Check train_data folder.")

	rnn_data = np.load('train_data/pg8800_rnn_char_data.npz')
	train_input = rnn_data['train_input']
	train_output = rnn_data['train_output']
	test_input = rnn_data['test_input']
	test_output = rnn_data['test_output']

	print('Data Information')
	print('Train data: 8,500 examples, Test data: 1,650 examples\nTime steps: 20, features: 38')
	return train_input, train_output, test_input, test_output