import numpy as np

def Sigmoid(x):
    return .5 * (1 + np.tanh(.5 * x))


def Sigmoid_grad(x):
    return Sigmoid(x) * Sigmoid(1 - x)


def accuracy(y, y_pred_class):
    return sum(y==y_pred_class) / len(y)

def fromOneHot(y):
    return np.argmax(y, axis=1)

def softmax(x):
    return np.exp(x) / np.expand_dims(np.sum(np.exp(x), axis=1),axis=1).repeat(x.shape[1], axis = 1)

def drop_out_matrices(layers_dims, num, keep_prob):
    D = {}
    for i in range(len(layers_dims)):
        D[i] = np.random.binomial(1, keep_prob[i], (num, layers_dims[i]))
    return D