import numpy as np
from load_dataset import load_dataset
from reshaper import reshaper
import logistic_model
'test'
path = "C:\\Project Slytherin\\Codes\\Coursera's Deep Learning Specialization\\Neural Networks and Deep Learning\\" \
       "Logistic Regression with a Neural Network Mindset\Dataset"
X_train, Y_train, X_test, Y_test, classes = load_dataset(path)
X_train_flatten, Y_train, X_test_flatten, Y_test = reshaper(X_train, Y_train, X_test, Y_test)

print(
    "The test set's original shape is: {0}. This means that we have {1} instances each of shape {2}, {2}, {3}. So we have"
    " a total of {4} pixels for each instance.".format(
        X_train.shape, X_train.shape[0], X_train.shape[1], X_train.shape[3],
        (X_train.shape[1] ** 2) * X_train.shape[3]))

X_train = X_train_flatten / 255
X_test = X_test_flatten / 255

params, grads, costs = logistic_model.train_model(num_pixels=X_train_flatten.shape[0], num_iterations=2000,
                                                  learning_rate=0.01, inputs=X_train,
                                                  labels=Y_train, print_cost=True)

predictions_train = logistic_model.predict(params, X_train)
predictions_test = logistic_model.predict(params, X_test)

print("train accuracy: {} %".format(100 - np.mean(np.abs(predictions_train - Y_train)) * 100))
print("test accuracy: {} %".format(100 - np.mean(np.abs(predictions_test - Y_test)) * 100))
