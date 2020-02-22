import numpy as np
from sigmoid_fuction import sigmoid, sigmoid_backward
from relu_function import relu, relu_backward


def initialize_parameters(layer_dims):
    # Arguments:
    #     layer_dims: dimensions of the network's layers
    # Returns:
    #     parameters: initialized weights and biases of each layer
    # Implements:
    #     it just initializes the parameters of network's layers

    parameters = {}  # The dictionary that's going to contain the weights and biases of network's layers
    L = len(layer_dims)

    for l in range(1, L):
        parameters['W{0}'.format(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * 0.01
        parameters['b{0}'.format(l)] = np.zeros((layer_dims[l], 1))

    return parameters


def linear_forward(A, W, b):
    # Arguments:
    #     A: result of the activation of the previous layer
    #     W: weights of the current layer
    #     b: bias of the current layer
    # Returns:
    #     Z: result of operations done on the input arguments
    #     cache: a tuple containing (A,W,b). we're gonna need them for backprop
    # Implements:
    #     a single forward pass

    Z = np.dot(W, A) + b
    cache = (A, W, b)

    return Z, cache


def linear_activation_forward(A_prev, W, b, activation):
    # Arguments:
    #     A_prev: the previous layer's activation value
    #     W: weights of the current layer
    #     b: bias of the current layer
    #     activation: a string which'll specify which activation function to use for this layer
    # Returns:
    #     A: the result of a single forward pass given the current arguments
    #     cache: a tuple containing the linear cache (A,W,b) and activation cache (Z). we'll need these for backprop
    # Implements:
    #     A single forward pass given the given activation function and arguments

    if activation == 'sigmoid':
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z), Z
    elif activation == 'relu':
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z), Z

    cache = (linear_cache, activation_cache)

    return A, cache


def L_model_forward(X, parameters):
    # Arguments:
    #     X: networks inputs
    #     parameters: dictionary containing network's parameters
    # Returns:
    #     AL: the network's output given the input and parameters
    #     caches: a list containing the cache of each layer's calculations. we'll need them for backprop
    # Implements:
    #     A complete forward pass of inputs through the network

    caches = []
    A = X
    L = len(parameters) // 2

    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters['W{0}'.format(l)], parameters['b{0}'.format(l)],
                                             activation='relu')
        caches.append(cache)
    AL, cache = linear_activation_forward(A, parameters['W{0}'.format(L)], parameters['b{0}'.format(L)],
                                          activation='sigmoid')

    caches.append(cache)

    return AL, caches


def compute_cost(AL, Y):
    # Arguments:
    #     AL: the output of the network
    #     Y: true labels of the inputs
    # Returns:
    #     cost: the cost of this iteration's calculations
    # Implements:
    #     network's cost function

    m = Y.shape[0]
    Y = Y.reshape(AL.T.shape)
    cost = (-1 / m) * np.sum(np.dot(Y, np.log(AL)) + np.dot((1 - Y), np.log(1 - AL)))
    cost = np.squeeze(cost)

    return cost


def linear_backward(dZ, cache):
    # Arguments:
    #     dZ: activation's derivative
    #     cache: a tuple containing previous layer's activation, weights and bias
    # Returns:
    #     dA_prev: A_prev's derivative
    #     dW: weights' derivative
    #     db: bias's derivative
    # Implements:
    #     a single backprop pass

    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = (1 / m) * np.dot(dZ, A_prev.T)
    db = (1 / m) * np.sum(dZ)
    dA_prev = np.dot(W.T, dZ)

    return dA_prev, dW, db


def linear_activation_backward(dA, cache, activation):
    # Arguments:
    #     dA: derivative of the activation
    #     cache: self explanatory
    #     activation: name of layer's activation function for calculating the right dZ
    # Returns:
    #     dA_prev: derivative of the previous layer's activation
    #     dW: layer's weights derivative
    #     db: layer's bias derivative
    # Implements:
    #     a single backward pass with regards of the layer's activation function

    linear_cache, activation_cache = cache

    if activation == 'relu':
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    elif activation == 'sigmoid':
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    return dA_prev, dW, db


def L_model_backward(AL, Y, caches):
    # Arguments:
    #     AL: the network's output
    #     Y: true labels
    #     caches: network's calculations
    # Returns:
    #     grads: the gradients calculated by backpropagation
    # Implements:
    #     backpropagation

    grads = {}
    L = len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    current_cache = caches[L - 1]
    grads["dA" + str(L - 1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL,
                                                                                                      current_cache,
                                                                                                      activation='sigmoid')

    for l in reversed(range(L - 1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads['dA' + str(l + 1)], current_cache,
                                                                    activation='relu')
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads


def update_parameters(parameters, grads, learning_rate):
    # Arguments:
    #     parameters: weights and biases of each layer
    #     grads: gradients calculated by backprop
    #     learning_rate: learning rate. you don't say!!!
    # Returns:
    #     updated parameters
    # Implements:
    #     updating network's parameters according to backprop calculations

    L = len(parameters) // 2

    for l in range(L):
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * grads['dW' + str(l + 1)]
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * grads['db' + str(l + 1)]

    return parameters


def train(inputs, labels, learning_rate, layers_dims, num_iterations, print_cost=False):
    # Arguments:
    #     inputs: network's inputs
    #     labels: labels of the inputs
    #     learning_rate:....
    #     layers_dims: dimensions of networks architecture
    #     num_iterations: number of iterations for the network's training
    #     print_cost: if true prints cost of network's calculations after each 100 iterations
    # Returns:
    #     parameters: network's parameters after training
    # Implements:
    #     network's training given the arguments

    X = inputs
    Y = labels
    parameters = initialize_parameters(layers_dims)

    for i in range(num_iterations):
        AL, caches = L_model_forward(X, parameters)
        cost = compute_cost(AL, Y)
        grads = L_model_backward(AL, Y, caches)
        parameters = update_parameters(parameters, grads, learning_rate)
        if print_cost and i % 100 == 0:
            print(cost)

    return parameters


def predict(X, parameters):
    # Arguments:
    #     X: inputs
    #     parameters: trained network's parameters
    # Returns:
    #     networks predictions according to the given arguments
    # Implements:
    #     a single forward pass with the trained parameters

    prediction, cache = L_model_forward(X, parameters)

    return prediction
