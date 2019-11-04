Project1:
# Road and bridge inspection and loss assessment system of UAV Based on CNN

路桥病害探测系统 

（1）Making road disease data set 

（2）Image preprocessing

（3）Make label: make the corresponding label of the preprocessed image as the label of the target detection image.

（4）Training model: using single SSD detection algorithm to detect road diseases, identify and classify the diseases

（5）Real time transmission of UAV image: the video image acquired by UAV is transmitted to the server in real time. The server uses the algorithm to segment the video and get the result of disease identification

（6）Automatic tracking of UAV: the UAV flies according to the established route, and detects and analyzes the flying over the road

Project2:
# Gradation and staging prediction of bladder tumor based on deep learning

基于深度学习的膀胱肿瘤分级和分期预测诊断
 
data processing:
（1）Training set, test set, verification set division
      We trained and tested the model in the Cdata dataset given by the topic. There were 1320 tumor images in different datasets. We chose 70% of the original dataset as the training set, 15% as the validation set, and 15% as the test set. 
（2）Data enhancement：
      Due to the translation, size and illuminance of the convolutional neural network, the neural network learning is irrelevant.
Features, enhanced model generalization, we performed random translation, random rotation, Gaussian blur, for each image in the dataset.
Random cropping, random noise processing, 5 processing, each image is expanded to 25, and then 25 images are watered
Flat flip, expand to 50, and randomly convert the 50 images into gamma, and finally expand the data set to 100 times.
（3）In order to maintain the consistency of the order of the data in the dataset and the order of the data in csv, the data is self-contained.Define tagging from zz_100000 to zz_231999 before the tag of the original data

Algorithm framework：
    We use convolutional neural networks for feature extraction for the following reasons:
The features learned by the Convolutional Neural Network are translationally invariant, and only a small number of training samples are needed to learn the data representation with generalization ability.
The 2 convolutional neural network can learn the spatial hierarchy of the pattern. The first convolutional layer will learn the smaller local pattern, and the second convolutional layer will learn the larger pattern composed of the first layer of features. analogy. This allows convolutional neural networks to effectively learn more and more complex and increasingly abstract visual concepts. In this question, we use deep convolutional neural networks to learn the deep features of medical images, which contribute to the accuracy of prediction.
