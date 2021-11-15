from Layer import Layer
import numpy as np


class DenseLayer:
    def __init__(self, layer_dimension, previous_layer_dimension):
        self.W_matrix = np.random.randn(layer_dimension, previous_layer_dimension) * 0.01  # weight matrix
        self.b_vector = np.zeros((layer_dimension, 1))  # bias vector

        self.dvW_matrix = None  # gradient for weights matrix
        self.dvb_matrix = None  # gradient for biases vector

        self.A_matrix = None  # cache with values after applying activation function
        self.Z_matrix = None  # cache with linear propagation values for current layer
        self.activation = None  # Activation object containing both activation function and its derivative
        self.dvA_matrix = None  # derivation of the Activation matrix
        self.dvZ_matrix = None  # derivation of the linear propagation matrix

    def linear_step(self, prev_activation):
        self.Z_matrix = np.dot(self.W_matrix, prev_activation) + self.b_vector
        return self.Z_matrix

    def activation_step(self):
        self.A_matrix = self.activation.activation_function(self.Z_matrix)

    def backward_step(self, m_train):
        m_inverse = 1 / m_train
        self.dvZ_matrix = self.dvA_matrix * self.activation.derivative(self.Z_matrix)
        self.dvW_matrix = np.dot(self.dvZ_matrix, self.A_matrix.T) * m_inverse
        self.dvb_matrix = np.sum(self.dvZ_matrix, axis=1, keepdims=True)

    def backward_activation(self):
        return np.dot(self.W_matrix.T, self.Z_matrix)
