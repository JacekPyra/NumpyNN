import numpy as np


class Dropout:
    def __init__(self, threshold):
        self.name = "Dropout"
        self.trainable = False
        self.threshold = threshold
        self.dropout = None
        self.prev_activation = None
        self.dvA_matrix = None

    def linear_step(self, prev_activation):
        # Create matrix with random continous uniform distribution
        # with the same dimensions as activation matrix from the previous layer.
        self.dropout = np.random.random(prev_activation.shape)
        # Convert dropout matrix to matrix of zeros and ones conditioned by set threshold
        self.dropout = np.where(self.dropout < self.threshold, 1, 0)
        self.prev_activation = prev_activation

    def activation_step(self):
        # shut down some random activations with current random dropout
        return self.prev_activation * self.dropout

    def backward_step(self, dvA_matrix, m_train):
        self.dvA_matrix = dvA_matrix

    def backward_activation(self):
        # shut down random activations with the same current random dropout
        return self.dvA_matrix * self.dropout

    def update_weights(self, learning_rate):
        pass
