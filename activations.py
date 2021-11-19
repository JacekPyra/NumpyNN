import numpy as np


# Sigmoid ##

class Sigmoid:
    def __init__(self):
        self.name = "Sigmoid"

    @staticmethod
    def activation_function(x):
        return 1 / (1 + (np.exp(-x)))

    def derivative(self, x):
        return self.activation_function(x) * (1 - self.activation_function(x))


# ReLU ##

class ReLU:
    def __init__(self):
        self.name = "ReLU"

    @staticmethod
    def activation_function(x):
        return np.maximum(0, x)

    @staticmethod
    def derivative(x):
        return np.heaviside(x, 1)


# ReLU ##

class Tanh:
    def __init__(self):
        self.name = "Tanh"

    @staticmethod
    def activation_function(x):
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
    @staticmethod
    def derivative(x):
        return 1 - ((np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))) ** 2


class LeakyReLU:
    def __init__(self):
        self.name = "LeakyReLU"

    @staticmethod
    def activation_function(x):
        return np.maximum(0.01 * x, x)
    @staticmethod
    def derivative(x):
        alpha = 0.01
        dx = np.ones_like(x)
        dx[x < 0] = alpha
        return dx


class Softmax:
    def __init__(self):
        self.name = "Softmax"

    @staticmethod
    def activation_function(x):
        exps = np.exp(x - np.max(x))
        div = exps / np.sum(exps, axis=0)
        return div

    @staticmethod
    def derivative(z):
        # Softmax layer is used together with Multi Class Binary Classification
        # and uses just its derivative to perform backpropagation.
        # To keep the backpropagation code consistent the derivative just returns 1
        return 1
