import numpy as np
from sigmoid_fuction import sigmoid


def initialize_neuron_parameters(number_of_pixels):
    # Arguments:
    #     number of pixels: the entire number of pixels for each image: height*width*num_channels
    # Returns:
    #     W: weight matrix initialized with zeros
    #     b: bias
    # Implements:
    #     Really? :|

    W = np.zeros((number_of_pixels, 1))  # weight matrix
    b = 0  # bias

    return W, b


def propagate(w, b, inputs, labels):
    # Arguments:
    #     w: weights of the model
    #     b: bias of the model
    #     inputs: self explanatory
    #     labels: self explanatory
    # Returns:
    #     grads: gradients for model's weights and bias
    #     cost: this round of propagation's cost
    # Implements:
    #     A single logistic function pass over the input data

    m = inputs.shape[1]  # number of instances
    A = sigmoid(np.dot(w.T, inputs) + b)  # activation value
    cost = (-1 / m) * np.sum(np.dot(labels, np.log(A).T) + np.dot(1 - labels, np.log(1 - A).T))  # cost value
    dw = (1 / m) * np.dot(inputs, (A - labels).T)  # derivative for w
    db = (1 / m) * np.sum(A - labels)  # derivative for b
    cost = np.squeeze(cost)
    grads = {'dw': dw, 'db': db}

    return grads, cost


def optimize(w, b, inputs, labels, num_iterations, learning_rate, print_cost=False):
    # Arguments:
    #     w: model's weights
    #     b: model's bias
    #     inputs: input data
    #     labels: labels
    #     num_iterations: number of iterations for the model to be trained on
    #     learning_rate: model's learning rate
    #     print_cost: a boolean value. determines whether should it print the cost after every 100 iterations or not
    # Returns:
    #     params: a dictionary containing model's weights and bias after being trained
    #     grads: gradients for model's weights and bias after being trained
    # Implements:
    #     The entire training phase for the given number of iterations

    costs = []

    if not print_cost:
        print('The model is training ......')

    for i in range(num_iterations):
        grads, cost = propagate(w, b, inputs, labels)
        dw = grads['dw']
        db = grads['db']
        w = w - learning_rate * dw
        b = b - learning_rate * db

        if i % 100 == 0:
            costs.append(cost)
        if print_cost and i % 100 == 0:
            print("Cost at iteration {0} is: {1}".format(i, cost))

    params = {'w': w, 'b': b}
    grads = {'dw': dw, 'db': db}

    return params, grads, costs


def train_model(num_pixels, num_iterations, learning_rate, inputs, labels, print_cost):
    # Arguments:
    #     num_pixels: the entire number of pixels for every instance
    #     The rest have been explained above
    # Returns:
    #     Unicorns! Yea, Unicorns. what do you think it returns dumbass?
    # Implements:
    #     This training yourself in writing documents for your functions is getting tiring.

    w, b = initialize_neuron_parameters(num_pixels)
    params, grads, costs = optimize(w, b, inputs, labels, num_iterations, learning_rate, print_cost)

    return params, grads, costs


def predict(params, pred_inputs):
    # Arguments:
    #     params: a dictionary containing model's weights and bias after training
    #     pred_inputs: the list of instances on the test set which we want to predict their labels
    # Returns:
    #     Y_prediction: model's predictions
    # Implements:
    #     a sigmoid function over pred_inputs with model's trained parameters

    w = params['w']
    b = params['b']
    A = sigmoid(np.dot(w.T, pred_inputs) + b)
    Y_prediction = np.zeros((1, pred_inputs.shape[1]))
    
    for i in range(A.shape[1]):
        if A[0][i] <= 0.5:
            Y_prediction[0][i] = 0
        else:
            Y_prediction[0][i] = 1

    return Y_prediction
