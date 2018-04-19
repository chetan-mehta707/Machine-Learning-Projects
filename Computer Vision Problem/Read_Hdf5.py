import tables
import numpy as np

hdf5_path = 'Dataset/dataset.hdf5'
subtract_mean = False

# open the hdf5 file
hdf5_file = tables.open_file(hdf5_path, mode='r')

# subtract the training mean
if subtract_mean:
    mm = hdf5_file.root.train_mean[0]
    mm = mm[np.newaxis, ...]

# Total number of samples
data_num = hdf5_file.root.train_img.shape[0]
