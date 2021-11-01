import glob
import os
import sys
from os import listdir

import h5py
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model


# Beluga
# direc = "/Users/sabrinaberger/CosmicDawn/GANning/z=8/"
# Local
#direc_plots = '/Users/sabrinaberger/CosmicDawn/'


def load_binary_data(filename, dtype=np.float32):
    """
    We assume that the data was written
    with write_binary_data() (little endian).

    Credit: Michael (CosmicDawn Group at McGill)
    """
    f = open(filename, "rb")
    data = f.read()
    f.close()
    _data = np.frombuffer(data, dtype)
    if sys.byteorder == 'big':
        _data = _data.byteswap()
    return _data


def load_HDF5(data_dir, z, file_type, dataset_name="xH_box"):
    """
    Loads HDF5 files and uses them as training set
    """
    train_arr = []
    data_dir_files = glob.glob(data_dir + file_type + "_*.h5")
    path = os.path.join(data_dir_files[0])
    hf = h5py.File(path, 'r')
    print("user_params " + str(hf["user_params"]))
    # print("size {}".format(hf.attrs["size"]))


    if file_type == "InitialConditions":
        path = os.path.join(data_dir_files[0])
        hf = h5py.File(path, 'r')
        # print(hf.attrs)
        data = hf[(file_type + "/" + dataset_name)]
        size = data.shape
        # print("initial {}".format(size))
        train_arr.append(data)
        # print("initial conditions array size is {}".format(np.shape(train_arr)))
        return train_arr

    for file in data_dir_files:
        path = os.path.join(file)
        hf = h5py.File(path, 'r')
        data = hf[(file_type + "/" + dataset_name)]
        size = data.shape
        print("size of cube is {}".format(size))
        z_data = hf.attrs["redshift"]
        if int(z_data) == z:
            train_arr.append(data)
            return train_arr
    print("Redshift not found.")
    return


def shape_data(dimension, data):
    data = np.reshape(np.asarray(data), (dimension, dimension, dimension))
    return data


# reduces size of data stored in the DataFeeder
def slice_selector(x, size, increment=10, start=0):
    '''
    Converts input cube of type np.ndarray into a list of numpy arrays representing
    necessary (uncorrelated) cube slices
    Credit: Sam
    '''
    # the number of maps is four times the number of map cubes passed into the method
    # this is at the current sampling rate of four maps per cube
    cube_slices1 = [x[i, :, :] for i in range(start, size-1, increment)]
    cube_slices2 = [x[:, i, :] for i in range(start, size-1, increment)]
    cube_slices3 = [x[:, :, i] for i in range(start, size-1, increment)]
    cube_slices = cube_slices1 + cube_slices2 + cube_slices3
    return cube_slices


def load_data_first(direc, direc_plots, redshift, filetype, type, test=False, size=200, binary=False, normalization=False):
    filenames = [direc + filename for filename in listdir(direc)]
    if binary:
        shaped_data = [shape_data(size, load_binary_data(data)) for data in filenames]
    else:
        shaped_data = load_HDF5(direc, redshift, filetype, type)

    if test == True:
        train_data = [slice_selector(shape, 40, 4, start=1) for shape in shaped_data] # avoiding first element
    else:
        train_data = [slice_selector(shape, size) for shape in shaped_data]

    old_shape = np.shape(train_data)
    train_data = np.asarray(train_data).reshape(old_shape[1] * old_shape[0], old_shape[2], old_shape[3])[1:]
    i = 0
    for train in train_data:
        plt.imshow(train)
        # plt.title("({}) ({})".format(filetype, type))
        plt.savefig(direc_plots + "{}".format(i))
        plt.close()
        i += 1
    if normalization is True:
        return normalization(train_data, np.amax(train_data), np.amin(train_data))

    return train_data
