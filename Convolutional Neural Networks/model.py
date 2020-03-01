import numpy as np
import h5py
import matplotlib.pyplot as plt


def zero_pad(X, pad):
    X_pad = np.pad(X, ((0, 0), (pad, pad), (pad, pad), (0, 0)), mode='constant', constant_values=0)

    return X_pad


def conv_single_step(a_slice_prev, W, b):
    s = a_slice_prev * W
    Z = np.sum(s)
    Z = Z + float(b)

    return Z


def conv_forward(A_prev, W, b, hyper_parameters):
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    (f, f, n_C_prev, n_C) = W.shape
    stride = hyper_parameters['stride']
    pad = hyper_parameters['pad']

    n_H = int(np.floor(((n_H_prev - f) + 2 * pad) / stride)) + 1
    n_W = int(np.floor(((n_W_prev - f) + 2 * pad) / stride)) + 1

    Z = np.zeros((n_H, n_W))

    A_prev_pad = zero_pad(A_prev, pad)

    for i in range(m):
        a_prev_pad = A_prev_pad[i]
        for h in range(n_H):
            vert_start = h
            vert_end = h + n_H

            for w in range(n_W):
                horiz_start = w
                horiz_end = w + n_W

                for c in range(n_C):
                    a_slice_prev = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]
                    weights = W[vert_start:vert_end, horiz_start:horiz_end, c, :]
                    biases = b[vert_start:vert_end, horiz_start:horiz_end, c, :]
                    Z[i, h, w, c] = conv_single_step(a_slice_prev, weights, biases)

    cache = (A_prev, W, b, hyper_parameters)
    return Z, cache


def pool_forward(A_prev, hyper_parameters, mode='max'):
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    f = hyper_parameters['f']
    stride = hyper_parameters['stride']
    n_H = int(1 + (n_H_prev - f) / stride)
    n_W = int(1 + (n_W_prev - f) / stride)
    n_C = n_C_prev
    A = np.zeros((m, n_H, n_W, n_C))
    for i in range(m):
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    vert_start = h * stride
                    vert_end = vert_start + f
                    horiz_start = w * stride
                    horiz_end = horiz_start + f
                    a_prev_slice = A_prev[i, vert_start:vert_end, horiz_start:horiz_end, c]
                    if mode == "max":
                        A[i, h, w, c] = np.max(a_prev_slice)
                    elif mode == "average":
                        A[i, h, w, c] = np.mean(a_prev_slice)

    cache = (A_prev, hyper_parameters)
    return A, cache


def conv_backward(dZ, cache):
    (A_prev, W, b, hyper_parameters) = cache
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    (f, f, n_C_prev, n_C) = W.shape
    stride = hyper_parameters["stride"]
    pad = hyper_parameters["pad"]

    (m, n_H, n_W, n_C) = dZ.shape
    dA_prev = np.zeros((m, n_H_prev, n_W_prev, n_C_prev))
    dW = np.zeros((f, f, n_C_prev, n_C))
    db = np.zeros((1, 1, 1, n_C))
    A_prev_pad = zero_pad(A_prev, pad)
    dA_prev_pad = zero_pad(dA_prev, pad)

    for i in range(m):
        a_prev_pad = A_prev_pad[i]
        da_prev_pad = dA_prev_pad[i]
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    vert_start = h * stride

                    vert_end = vert_start + f
                    horiz_start = w * stride

                    horiz_end = horiz_start + f
                    a_slice = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]
                    da_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :] += W[:, :, :, c] * dZ[i, h, w, c]
                    dW[:, :, :, c] += a_slice * dZ[i, h, w, c]
                    db[:, :, :, c] += dZ[i, h, w, c]
        dA_prev[i, :, :, :] = da_prev_pad[pad:-pad, pad:-pad, :]
    return dA_prev, dW, db


def create_mask_from_window(x):
    return x == np.max(x)


def distribute_value(dZ, shape):
    (n_H, n_W) = shape
    average = dZ / (n_H * n_W)
    a = np.ones(shape) * average
    return a
