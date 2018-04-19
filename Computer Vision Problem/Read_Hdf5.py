import tables
import numpy as np
from random import shuffle
from math import ceil
import matplotlib.pyplot as plt

hdf5_path = '/Users/chetan/Documents/Git Projects/Machine Learning/Computer Vision Problem/Dataset/dataset.hdf5'  # address to where you want to save the hdf5 file
subtract_mean = True
batch_size = 50
nb_class = 2

# open the hdf5 file
hdf5_file = tables.open_file(hdf5_path, mode='r')

try:
    # subtract the training mean
    if subtract_mean:
        mm = hdf5_file.root.train_mean[0]
        mm = mm[np.newaxis, ...]
    
    # Total number of samples
    data_num = hdf5_file.root.train_img.shape[0]
    
    # create list of batches to shuffle the data
    batches_list = list(range(int(ceil(float(data_num) / batch_size))))
    shuffle(batches_list)
    
    # loop over batches
    for n, i in enumerate(batches_list):
        i_s = i * batch_size  # index of the first image in this batch
        i_e = min([(i + 1) * batch_size, data_num])  # index of the last image in this batch
        print('i_s:',i_s,' i_e:',i_e)
    
        # read batch images and remove training mean
        images = hdf5_file.root.train_img[i_s:i_e]
        print('len:',len(images))
        if subtract_mean:
            images -= mm
    
        # read labels and convert to one hot encoding
        labels = hdf5_file.root.train_labels[i_s:i_e]
        labels_one_hot = np.zeros((len(images), nb_class))
        labels_one_hot[np.arange(len(images)), labels] = 1
    
        print(n+1, '/', len(batches_list))
    
        print (labels[0], labels_one_hot[0, :])
        plt.imshow(images[0])
        plt.show()
    
        if n == 5:  # break after 5 batches
            break
finally:
    hdf5_file.close()

