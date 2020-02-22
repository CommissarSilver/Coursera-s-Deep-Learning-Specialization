import model
from load_dataset import load_dataset
import numpy as np
import matplotlib.pyplot as plt

dataset_path = "C:\\Project Slytherin\\Codes\\Coursera's Deep Learning Specialization\\Neural Networks and Deep Learning\\Building Your Deep Neural Network Step by Step\\Dataset"
X_train, Y_train, X_test, Y_test, classes = load_dataset(path=dataset_path)

X_train_flatten = X_train.reshape(X_train.shape[0], -1).T  # The "-1" makes reshape flatten the remaining dimensions
X_test_flatten = X_test.reshape(X_test.shape[0], -1).T

X_train = X_train_flatten / 255
X_test = X_test_flatten / 255
print(X_train.shape)
model_parameters = model.train(inputs=X_train, labels=Y_train, learning_rate=0.0075, layers_dims=[12288, 20, 7, 5, 1],
                               num_iterations=2500, print_cost=True)
test = model.predict(X=X_test, parameters=model_parameters)
print('hi')
