import costFunctions
import numpy as np


class Model:
    def __init__(self, cost_function):
        self.cost_function = cost_function()
        self.layers_list = []
        self.last_activation = None
        self.cost = []

    def sequential(self, layers):
        self.layers_list = layers

    def forward_propagation(self, X):
        prev_activation = X
        for layer in self.layers_list:
            layer.linear_step(prev_activation)
            prev_activation = layer.activation_step()
        self.last_activation = prev_activation

    def backward_propagation(self, Y, m_train):
        last_activation = costFunctions.CrossEntropy.derivative(self.last_activation, Y)
        for layer in reversed(self.layers_list):
            layer.backward_step(last_activation, Y.shape[1])
            last_activation = layer.backward_activation()

    def update_weights(self, learning_rate):
        for layer in self.layers_list:
            layer.update_weights(learning_rate)

    def calculate_cost(self, Y):
        return self.cost_function.calculate_cost(self.last_activation, Y)

    def train(self, X, Y, learning_rate, iterations_count):
        percentage = iterations_count / 100
        for i in range(iterations_count):
            self.forward_propagation(X)
            self.cost.append(self.calculate_cost(Y))
            self.backward_propagation(Y, Y.shape[1])
            self.update_weights(learning_rate)

            if i % percentage == 0:
                ip = i / percentage
                print('\rProgress: [%d%%]' % ip, end="")
                print(" , Loss: " + str(self.cost[-1]), end="")

        print("")
        print(self.cost[::10])

    def evaluate(self, test_data, test_labels):
        self.forward_propagation(test_data)
        predictions = np.round(self.last_activation)
        print(predictions.shape)

        predictions = np.argmax(predictions, axis=0)
        test_labels = np.argmax(test_labels, axis=0)

        print("Evaluation percentage: ", np.mean(predictions == test_labels) * 100)

        print("Exemplary predictions: ", predictions[:10])
        print("Correct answers:       ", test_labels[:10])
