import numpy as np


def relu(Z):
    return np.maximum(0, Z)


def relu_backward(dA, cache):
    Z = cache
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    return dZ
