from random import shuffle
import glob
import numpy as np
import tables
shuffle_data = True  # shuffle the addresses before saving
hdf5_path = 'Dataset/dataset.hdf5'  # address to where you want to save the hdf5 file
cat_dog_train_path = 'Dataset/*.jpg'

# read addresses and labels from the 'Dataset' folder
addrs = glob.glob(cat_dog_train_path)
labels = [0 if 'hole' in addr else 1 for addr in addrs]  # 0 = Hole, 1 = dirt

# to shuffle data
if shuffle_data:
    c = list(zip(addrs, labels))
    shuffle(c)
    addrs, labels = zip(*c)

# Divide the hata into 60% train, 20% validation, and 20% test
train_addrs = addrs[0:int(0.6*len(addrs))]
train_labels = labels[0:int(0.6*len(labels))]

val_addrs = addrs[int(0.6*len(addrs)):int(0.8*len(addrs))]
val_labels = labels[int(0.6*len(addrs)):int(0.8*len(addrs))]

test_addrs = addrs[int(0.8*len(addrs)):]
test_labels = labels[int(0.8*len(labels)):]

data_order = 'tf'  # 'th' for Theano, 'tf' for Tensorflow
img_dtype = tables.UInt8Atom()  # dtype in which the images will be saved

# check the order of data and chose proper data shape to save images
if data_order == 'th':
    data_shape = (0, 3, 224, 224)
elif data_order == 'tf':
    data_shape = (0, 224, 224, 3)

# open a hdf5 file and create earrays
hdf5_file = tables.open_file(hdf5_path, mode='w')

train_storage = hdf5_file.create_earray(hdf5_file.root, 'train_img', img_dtype, shape=data_shape)
val_storage = hdf5_file.create_earray(hdf5_file.root, 'val_img', img_dtype, shape=data_shape)
test_storage = hdf5_file.create_earray(hdf5_file.root, 'test_img', img_dtype, shape=data_shape)

mean_storage = hdf5_file.create_earray(hdf5_file.root, 'train_mean', img_dtype, shape=data_shape)

# create the label arrays and copy the labels data in them
hdf5_file.create_array(hdf5_file.root, 'train_labels', train_labels)
hdf5_file.create_array(hdf5_file.root, 'val_labels', val_labels)
hdf5_file.create_array(hdf5_file.root, 'test_labels', test_labels)

# a numpy array to save the mean of the images
mean = np.zeros(data_shape[1:], np.float32)

# loop over train addresses
for i in range(len(train_addrs)):
    # print how many images are saved every 1000 images
    if i % 1000 == 0 and i > 1:
        print 'Train data: {}/{}'.format(i, len(train_addrs))

    # read an image and resize to (224, 224)
    # cv2 load images as BGR, convert it to RGB
    addr = train_addrs[i]
    img = cv2.imread(addr)
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # add any image pre-processing here

    # if the data order is Theano, axis orders should change
    if data_order == 'th':
        img = np.rollaxis(img, 2)

    # save the image and calculate the mean so far
    train_storage.append(img[None])
    mean += img / float(len(train_labels))

# loop over validation addresses
for i in range(len(val_addrs)):
    # print how many images are saved every 1000 images
    if i % 1000 == 0 and i > 1:
        print 'Validation data: {}/{}'.format(i, len(val_addrs))

    # read an image and resize to (224, 224)
    # cv2 load images as BGR, convert it to RGB
    addr = val_addrs[i]
    img = cv2.imread(addr)
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # add any image pre-processing here

    # if the data order is Theano, axis orders should change
    if data_order == 'th':
        img = np.rollaxis(img, 2)

    # save the image
    val_storage.append(img[None])

# loop over test addresses
for i in range(len(test_addrs)):
    # print how many images are saved every 1000 images
    if i % 1000 == 0 and i > 1:
        print 'Test data: {}/{}'.format(i, len(test_addrs))

    # read an image and resize to (224, 224)
    # cv2 load images as BGR, convert it to RGB
    addr = test_addrs[i]
    img = cv2.imread(addr)
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # add any image pre-processing here

    # if the data order is Theano, axis orders should change
    if data_order == 'th':
        img = np.rollaxis(img, 2)

    # save the image
    test_storage.append(img[None])

# save the mean and close the hdf5 file
mean_storage.append(mean[None])
hdf5_file.close()
