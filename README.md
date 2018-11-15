# Code for IJCNN 2018 submission
This repository contains the tensorflow implementation for the paper: "Emotion Recognition from Multi-Channel EEG through Parallel Convolutional Recurrent Neural Network"
## About the paper
* Title: [Emotion Recognition from Multi-Channel EEG through Parallel Convolutional Recurrent Neural Network](https://ieeexplore.ieee.org/document/8489331)
* Authors: [Yilong Yang](https://ynulonger.github.io/), Qingfeng Wu, Ming Qiu, Yingdong Wang, Xiaowei Chen
* Institution: Xiamen University
* Published in: 2018 International Joint Conference on Neural Networks (IJCNN) 
* DOI: 10.1109/IJCNN.2018.8489331
## Instructions
^ 1. Before running the code, please download the DEAP dataset, unzip it and place it into the right directory. The dataset can be found [here](http://www.eecs.qmul.ac.uk/mmv/datasets/deap/index.html). Each .mat data file contains the EEG signals and consponding labels of a subject. There are 2 arrays in the file: **data** and **labels**. The shape of **data** is (40, 40, 8064). The shape of **label** is (40,4). Each .pkl file contains a numpy.ndarray variable. It stores the pre_processed data with the shape of (segments, window_size, width, height), in this paper, it is (2400,128,9,9).
^ 2. Please run the deap_pre_process.py to Load the origin .mat data file and transform it into .pkl file.
^ 3. Using cv.py to train and test the model (using 10-fold cross-validation), result of each fold will be saved in a .xls file.
^ 4. count_accuracy.py is used to caculate the final accuracy of the model.
## Requirements
+ Pyhton 3
+ scipy
+ numpy
+ pandas
+ pickle
+ sk-learn
+ pickle
+ tensorflow (1.4 version)
+ import xlrd
+ import xlwt

If you have any questions, please contact yilongyang@stu.xmu.edu.cn