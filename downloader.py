'''
Download train and test data.

                                                                    Written by Hyungwon Yang
                                                                                2016. 04. 17
                                                                                    EMCS Lab
This script downloads the train and test dataset.                                                       
Please run this script in the HY_python_NN folder.
Do not move this script to other directory.
'''

# This script downloads the train and test dataset.
# Please run this script in the HY_python_NN folder.
# Do not move this script to other directory.



import sys
import requests

# Check argument.
if len(sys.argv) != 2:
    print("Input arguments are incorrectly provided. File name should be assigned.")
    print("*** USAGE ***")
    print("Ex. python downloader.py $filename")
    print("*** DATA LIST ***")
    print("1. mnist : MNIST data. classification")
    print("2. body : body data from matlab. regression")
    print("3. building : building data from matlab. regression")
    print("4. artandacou : articulation and acoustics data. regression.")
    print("5. cancer : cancer data from matlab. classification")
    print("6. pg8800_word : The part of Project Gutenberg's The Divine Comedy, Complete ebook data. splited into words. classification")
    print("7. pg8800_char : The part of Project Gutenberg's The Divine Comedy, Complete ebook data. splited into characters. classification")
    raise ValueError('RETURN')

data = sys.argv[1]


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


if len(sys.argv) != 2:
    print("Input arguments are incorrectly provided. One argument should be assigned.")
    print("1. File Name.")
    print("*** USAGE ***")
    print("Ex. python downloader.py $filename")
    print("*** DATA LIST ***")
    print("1. mnist : MNIST data. classification")
    print("2. body : body data from matlab. regression")
    print("3. building : building data from matlab. regression")
    print("4. artandacou : articulation and acoustics data. regression.")
    print("5. cancer : cancer data from matlab. classification")
    print("6. pg8800_word : The part of Project Gutenberg's The Divine Comedy, Complete ebook data. splited into words. classification")
    print("7. pg8800_char : The part of Project Gutenberg's The Divine Comedy, Complete ebook data. splited into characters. classification")
    raise ValueError('RETURN')

data = sys.argv[1]

if data == 'mnist':

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

elif data == 'body':

    print("Downloading bodyData...")
    datafile = "0B9lwe_GFwe2oV2lhbS1PNFdKbnM"
    savefile = "train_data/bodyData.mat"
    download_file_from_google_drive(datafile, savefile)

    print("Dataset Downloaded successfully. Check train_data folder.")

elif data == 'building':

    print("Downloading buildingData...")
    datafile = "0B9lwe_GFwe2oMmxPalZSY1pEMkU"
    savefile = "train_data/buildingData.mat"
    download_file_from_google_drive(datafile, savefile)

    print("Dataset Downloaded successfully. Check train_data folder.")

elif data == 'artandacou':

    print("Downloading new articulation and acoustics...")

    print("Downloading new_articulation...")
    datafile = "0B9lwe_GFwe2oMkFVdzE1QXhOdG8"
    savefile = "train_data/new_articulation.pckl"
    download_file_from_google_drive(datafile, savefile)

    print("Downloading new_acoustics...")
    datafile = "0B9lwe_GFwe2oS2lTZG1oQ2ZCejQ"
    savefile = "train_data/new_acoustics.pckl"
    download_file_from_google_drive(datafile, savefile)

    print("Dataset Downloaded successfully. Check train_data folder.")

elif data == 'cancer':

    print("Downloading cancerData...")
    datafile = "0B9lwe_GFwe2oNEtSMHFXSWk3Rkk"
    savefile = "train_data/cancerData.mat"
    download_file_from_google_drive(datafile, savefile)

    print("Dataset Downloaded successfully. Check train_data folder.")

# elif data == 'pg8800_word':
#
#     print("Downloading pg8800...")
#     datafile =
#     savefile =
#     download_file_from_google_drive(datafile, savefile)
#
#     print("Dataset Downloaded successfully. Check train_data folder.")

elif data == 'pg8800_char':

    print("Downloading pg8800...")

    print("Downloading pg8800_ann_char_data.npz")
    datafile = "0B9lwe_GFwe2oeWJlMDJKdjg2RXM"
    savefile = "train_data/pg8800_ann_char_data.npz"
    download_file_from_google_drive(datafile, savefile)

    print("Downloading pg8800_lstm_char_data.npz")
    datafile = "0B9lwe_GFwe2oYlpfbEdMVllOSmc"
    savefile = "train_data/pg8800_lstm_char_data.npz"
    download_file_from_google_drive(datafile, savefile)

    print("Dataset Downloaded successfully. Check train_data folder.")

else:
    print(data,'is not present in the data list. Please type the data named below.')
    print("*** DATA LIST ***")
    print("1. mnist : MNIST data. classification")
    print("2. body : body data from matlab. regression")
    print("3. building : building data from matlab. regression")
    print("4. artandacou : articulation and acoustics data. regression.")
    print("5. cancer : cancer data from matlab. classification")
    print("6. pg8800_word : The part of Project Gutenberg's The Divine Comedy, Complete ebook data. splited into words. classification")
    print("7. pg8800_char : The part of Project Gutenberg's The Divine Comedy, Complete ebook data. splited into characters. classification")

