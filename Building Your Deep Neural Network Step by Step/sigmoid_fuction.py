import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoid_backward(dA, cache):
    Z = cache
    s = 1 / (1 + np.exp(-Z))
    dZ = dA * s * (1 - s)
    return dZ
