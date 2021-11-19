import numpy as np


class CrossEntropy:
    @staticmethod
    def calculate_cost(last_layer, Y_values):
        m = Y_values.shape[1]
        logprobs = np.multiply(np.log(last_layer), Y_values) + np.multiply(np.log(1 - last_layer), 1 - Y_values)
        return - np.sum(logprobs) / m

    @staticmethod
    def derivative(last_layer, Y_values):
        return - (np.divide(Y_values, last_layer) - np.divide(1 - Y_values, 1 - last_layer))


class MultiClassCrossEntropy:
    @staticmethod
    def calculate_cost(last_layer, Y_values):
        m = Y_values.shape[1]
        reg_sum = 0
        return np.sum((np.log(last_layer) * (-Y_values)) - (np.log(1 - last_layer) * (1 - Y_values))) / m + reg_sum

    @staticmethod
    def derivative(last_layer, Y_values):
        return last_layer - Y_values
