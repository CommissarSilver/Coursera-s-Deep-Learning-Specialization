import numpy as np


def reshaper(train_data, train_data_labels, test_data, test_data_labels):
    # Arguments:
    #     train_data: training data
    #     test_data: test data
    #     train_data_labels: labels for the training data
    #     test_data_labels: labels for the test data
    # Returns:
    #     train_data_flatten: flatted train data for training the model
    #     train_data_labels: transposed train data labels for training the model
    #     test_data_flatten: flatted test data for testing the model
    #     test_data_labels: transposed test data labels for testing the model
    # Implements:
    #     nothing really special. flattens both the training data and test data for training the logistic model.
    #     also transposes their labels as well.

    train_data_flatten = train_data.reshape(train_data.shape[0], -1).T
    test_data_flatten = test_data.reshape(test_data.shape[0], -1).T
    train_data_labels = train_data_labels.reshape(1, train_data_labels.shape[0])
    test_data_labels = test_data_labels.reshape(1, test_data_labels.shape[0])

    return train_data_flatten, train_data_labels, test_data_flatten, test_data_labels
