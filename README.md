# Speech-recognition-using-MFCC-STFT-and-Mixed-feature
* This repository using MFCC, STFT and their concatenated feature(mixed feature) to classify digits using CNN and ANN.

* The ***dataprocess.py*** will transvert the MFCC and STFT feature into unidimensional features or $[25,13]$ images where 25 means the 25 different filters and 13 means each raw data is divided into 13 frames.

* The ***CNN.py*** and ***ANN.py*** will use the images or vectors dataprocess.py generated to spilt train and test data, train the neural network and test its accuracy.

**if anyone are interested in the dataset, contact me via E-mail: Heba_private@163.com. The dataset contains 23 students ranged from 16-22speaking English digits 0-9 ten times per student and Chinese digits 0-9 ten times per student. Hence, there are 2300 English spoken digit signals and 2300 Chinese spoken digit signals**
