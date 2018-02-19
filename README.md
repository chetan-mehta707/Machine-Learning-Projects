# Machine-Learning-Projects
This project is developed by the learning I received while studying Machine Learning Concepts. We are trying to solve a simple classification problem here.

Problem Statement:

Dataset : 
We are generating a dataset as simple as each training example is having 2 co-ordinates(x,y) which lies in region with some mean value say(xm,ym). All the training examples are labeled with one of the 3 classes (0,1,2) for the means (xm1,ym1),(xm2,ym2) and (xm3,ym3) resp. I am using multi-variate normal form for generating the distribution.

I am using Numpy arrays for storing the features. All the labels are converted to one-hot encoding prior to processing.

Scikit-learn for generating test and train split.

matpotlib for data visualization.

All the required functions are derived within the script viz. Activation(),SoftMax(), CrossEntropy() which gives a better understanding of how the data will be processed.

I have given a tensorflow version for the same problem to spot the difference.
