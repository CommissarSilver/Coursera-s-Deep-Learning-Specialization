import h5py
import numpy as np


def load_dataset(path):
    # Arguments:
    #     path: path of where the dataset is
    # Returns:
    #     train_data: the training data of cat and non-cat
    #     train_data_labels: the training data labels
    #     test_data: the test data of cat and non=cat
    #     test_data labels: the test data labels
    #     classes: the classes of the dataset (cat and non-cat)
    # Implements:
    #     Reads the dataset h5 files in the dataset folder and returnes them seperated by training data and it's labels, test data and it's labels and the dataste classes

    train_data_h5 = h5py.File('{0}\\train_catvnoncat.h5'.format(path), 'r')
    test_data_h5 = h5py.File('{0}\\test_catvnoncat.h5'.format(path), 'r')

    train_data = train_data_h5['train_set_x']
    train_data_labels = train_data_h5['train_set_y']
    classes = train_data_h5['list_classes']

    test_data = test_data_h5['test_set_x']
    test_data_labels = test_data_h5['test_set_y']

    return np.array(train_data), np.array(train_data_labels), np.array(test_data), np.array(test_data_labels), classes
