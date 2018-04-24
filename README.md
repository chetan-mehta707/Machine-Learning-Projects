# Machine-Learning-Projects
This project is developed by the learning I received while studying Machine Learning Concepts.
Follwing are the problem statements I have identified.

## 1. Computer Vision Problem: Image Classifier 

Problem Statement:

Dataset : Image dataset of clothes with dirt and hole.
Here, I am trying to classify image data of clothes into couple of category(viz clothes with hole and dirt).
For gathering dataset I am using javascript and python to collect data from google image.
Search the image you want to collect from google image search, scroll down till you feel data is sufficient. Run web.js which will give you a urls.txt file containing url's for the images. Then using pyhton's request module we will store those images into a directory.

Once you have gathered a data the next step comes where you label your image, compress them and add it to a batch file. I am using OpenCv library and HDF5 format to store image data.

Once the data is filtered and processed its time to built our classifier, I am using CNN architecture for building classifer using Keras.


## 2. Classification Problem:

Problem Statement:

Dataset : 
We are generating a dataset as simple as each training example is having 2 co-ordinates(x,y) which lies in region with some mean value say(xm,ym). All the training examples are labeled with one of the 3 classes (0,1,2) for the means (xm1,ym1),(xm2,ym2) and (xm3,ym3) resp. I am using multi-variate normal form for generating the distribution.

I am using Numpy arrays for storing the features. All the labels are converted to one-hot encoding prior to processing.

Scikit-learn for generating test and train split.

matpotlib for data visualization.

All the required functions are derived within the script viz. Activation(),SoftMax(), CrossEntropy() which gives a better understanding of how the data will be processed.

I have given a tensorflow version for the same problem to spot the difference..

## 3. Regression Problem:

Problem Statement :

Dataset:
Dataset consist of various parameters considered for calculating Energy consumption. We need to build a model that accurately predicts the Energy consumption. You can find the dataset in the Regression Problem Folder. 
