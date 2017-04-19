# This is HY_python_NN. 
                                                              Hyungwon Yang
                                                                 2015.02.02
                                                                   EMCS lab


Machine learning toolbox based on Python and Tensorflow.



Linux and MacOSX (This script is not tested on Windows)
---

- Python 3.5 and 3.6
- tensorflow



CONTENTS
---
- Machine learning scripts for classification & regression problems.
1. main(dir) : Codes that build and run various neural networks are stored in this directory.
2. reports(dir): Exeperiments such as comparing between or among machine learning task performances or replicating other scholars' works related to machine learning tasks are implemented and reported in this directory.
3. train_data(dir): Small set of training and testing data (some of them are already preprocessed for convenience's sake) are ready for implementing simple experiments or testing newly structued networks.
4. downloader.py(script): It downloads some of the dataset into train_data directory. Read the list of downloadable dataset and save it to your local disk by simply running this script.
		
CONTACTS
---------------------------------------------------------------------------

Hyungwon Yang / hyung8758@gmail.com

- Advisor: Hosung Nam 
- Supporter: Heejo You, Jaekoo Kang


VERSION HISTORY
---------------------------------------------------------------------------
1.0 (2015.02.22)
 All 8 scripts were added. 
 - loadfile: It contains example files for running scripts.
 - ANN_model: This script constructs ANN and DBN networks. 
 Feedforward neural network and stochastic gradient descent.
 - SetValues: This class initiates all the default values including inputs, 
 outputs, weight and biases
 - errortools: Visual plotting and error calculation. 
 MSE, Pearson r, and correlation plot.
 - ANNforClass: ANN model to solve classification problem
 - ANNforCurvefit: ANN model to solve curve-fitting problem (regression)
 - DBNforClass: DBN model to solve classification problem
 - DBNforCurvefit: DBN model to solve curve-fitting problem (regression)

1.1 (2015.03.07)
 - In classification, prediction for accuracy bug fixed.

1.2 (2015.04.30)
 - ANN scripts that runnable in python2 is moved to 'VersionHistory' folder. (refer to for_python2.zip file)
 - Main scripts are re-written in order to run it in python3. From now on, ANN script is only supportable in python3.

2.1 (2017.04.14)
 - All of the scripts are rewritten based on tensorflow.
 - RNN(Vanilla, LSTM, GRU) is added.
 - Smoe reports are uploaded.
