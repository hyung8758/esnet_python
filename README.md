# Easy and Simple Neural nETwork: esnet (python version.)
                                                              Hyungwon Yang
                                                                 2015.02.02
                                                                   EMCS lab


### Machine learning toolbox based on Python and Tensorflow.
---

- This toolbox is easy to implement to your research. 
- Just focus on neural network parameters such as the number of hidden layers and units or activation functions. 
Then a main network based on given parameters will be automatically generated for a task.


### Version Compatibility
---

- Linux and MacOSX (This script is not tested on Windows)
- Python 3.5 and 3.6 (Python2 is not supported)
- Tensorflow 1.1

### CONTENTS
---
- Machine learning scripts for classification and regression problems.
 1. main(dir) : Codes that build and run various neural networks are stored in this directory.
 2. reports(dir): Experiments such as comparing between or among machine learning task performances or replicating other scholars' works related to machine learning tasks are implemented and reported in this directory.
 3. train_data(dir): Small set of training and testing data (some of them are already preprocessed for convenience's sake) are ready for implementing simple experiments or testing newly structured networks.
 4. train.py(script): This script shows how to build various neural networks and train and test them with diverse datasets. 
		
### CONTACTS
---------------------------------------------------------------------------

Should you have any question while using this toolbox, please do not hesitate to contact Hyungwon Yang. 
(bug reports or suggestions for developing the toolbox are always welcomed.)

- Developer: Hyungwon Yang (hyung8758@gmail.com)
- Advisor: Hosung Nam 
- Supporter: Heejo You, Jaekoo Kang


### VERSION HISTORY
---------------------------------------------------------------------------
1.0 (2015.02.22)
 All 8 scripts were added. 
 - loadfile: It contains example files for running scripts.
 - ANN_model: This script constructs ANN and DBN networks. 
 Feed-forward neural network and stochastic gradient descent.
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
 - Some reports are uploaded.

2.2 (2017.05.01)
 - Compatibility update with tensorflow version 1.1 
 - Downloading function is merged to loadfile.py so downloader.py is removed.
 - Minor bugs are fixed. 
 - During training session, network settings and processing information will be printed on the command line.
