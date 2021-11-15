import numpy as np
# Sigmoid ##

class Sigmoid:
    def __init__(self):
        pass

    def activation_function(self, x):
        return 1 / (1 + (np.exp(-x)))

    def derivative(self, x):
        return self.activation_function(x) * (1 - self.activation_function(x))


# ReLU ##

class ReLU:
    def __init__(self):
        pass

    def activation_function(self, x):
        return np.maximum(0, x)

    def derivative(self, x):
        return np.heaviside(x, 1)
