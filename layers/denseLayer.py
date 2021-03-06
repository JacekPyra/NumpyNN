import numpy as np


class DenseLayer:
    def __init__(self, input_layer_dimension, output_layer_dimension, activation, mean=None, std=None):
        # Dense layer
        # The default initialization is Xavier initialization.
        # When mean and standard deviation (std) are not set, std take the value of square root of inverse number
        # of the previous layer dimension. This initialization prevents the vanishing gradients and exploding gradients.
        if std is None:
            std = np.sqrt(1 / input_layer_dimension)
        if mean is None:
            mean = 0
        self.name = "Dense"
        self.trainable = True
        self.W_matrix = np.random.normal(mean, std, (output_layer_dimension, input_layer_dimension))  # weight matrix
        self.b_vector = np.zeros((output_layer_dimension, 1))  # bias vector

        self.dvW_matrix = None  # gradient for weights matrix
        self.dvb_vector = None  # gradient for biases vector

        self.A_matrix = None  # cache with values after applying activation function
        self.Z_matrix = None  # cache with linear propagation values for current layer
        self.activation = activation()  # Activation object containing both activation function and its derivative
        self.dvA_matrix = None  # derivation of the Activation matrix
        self.dvZ_matrix = None  # derivation of the linear propagation matrix
        self.A_prev_matrix = None

    def linear_step(self, prev_activation):
        self.A_prev_matrix = prev_activation
        self.Z_matrix = np.dot(self.W_matrix, prev_activation) + self.b_vector
        return self.Z_matrix

    def activation_step(self):
        self.A_matrix = self.activation.activation_function(self.Z_matrix)
        return self.A_matrix

    def backward_step(self, dvA_matrix, m_train):
        m_inverse = 1 / m_train
        self.dvA_matrix = dvA_matrix
        self.dvZ_matrix = self.dvA_matrix * self.activation.derivative(self.Z_matrix)
        self.dvW_matrix = np.dot(self.dvZ_matrix, self.A_prev_matrix.T) * m_inverse
        self.dvb_vector = np.sum(self.dvZ_matrix, axis=1, keepdims=True) * m_inverse

    def backward_activation(self):
        return np.dot(self.W_matrix.T, self.dvZ_matrix)

    def update_weights(self, learning_rate):
        self.W_matrix = self.W_matrix - learning_rate * self.dvW_matrix
        self.b_vector = self.b_vector - learning_rate * self.dvb_vector

    def get_weights(self):
        return self.W_matrix
