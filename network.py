import costFunctions
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
import math

class Model:
    def __init__(self, cost_function):
        self.cost_function = cost_function()
        self.layers_list = []
        self.last_activation = None
        self.cost = []

    def sequential(self, layers):
        self.layers_list = layers

    def forward_propagation(self, X, training=False):
        prev_activation = X
        for layer in self.layers_list:
            # layer.trainable refers to a single layer like Dropout which are not meant to be trained.
            # training refers to training operation (ADAM, RMSprop, etc).
            # if training = False then all untrainable layers (layer.trainable = False)
            # will not affect feedforwarding the data. All the data will depend on the parameters of the network.
            if not layer.trainable and not training:
                continue
            layer.linear_step(prev_activation)
            prev_activation = layer.activation_step()
        self.last_activation = prev_activation

    def backward_propagation(self, Y):
        delta = self.cost_function.derivative(self.last_activation, Y)
        for layer in reversed(self.layers_list):
            layer.backward_step(delta, Y.shape[1])
            delta = layer.backward_activation()

    def update_weights(self, learning_rate):
        for layer in self.layers_list:
            layer.update_weights(learning_rate)

    def calculate_cost(self, Y):
        return self.cost_function.calculate_cost(self.last_activation, Y)

    def Adam(self):
        pass
    def RMSprop(self):
        pass

    def mini_batch(self, X, Y, batch_size):
        m = Y.shape[1]  # number of training examples
        mini_batches = []

        permutation = list(np.random.permutation(m))
        shuffled_X = X[:, permutation]
        shuffled_Y = Y[:, permutation]
        inc = batch_size
        num_complete_minibatches = math.floor(m / batch_size) # number of mini batches of size mini_batch_size in your partitionning

        for k in range(0, num_complete_minibatches):
            mini_batch_X = shuffled_X[:, k * inc:(k + 1) * inc]
            mini_batch_Y = shuffled_Y[:, k * inc:(k + 1) * inc]
            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)

        # For handling the end case (last mini-batch < mini_batch_size i.e less than 64)
        if m % batch_size != 0:
            mini_batch_X = shuffled_X[:, num_complete_minibatches * inc:]
            mini_batch_Y = shuffled_Y[:, num_complete_minibatches * inc:]
            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)
        return mini_batches

    def train(self, learning_rate, iterations_count, X_train, Y_train, X_test, Y_test):
        percentage = iterations_count / 100
        previous_validation_loss = 100000000.0
        for i in range(iterations_count):
            self.forward_propagation(X_train, training=True)
            self.cost.append(self.calculate_cost(Y_train))
            self.backward_propagation(Y_train)
            self.update_weights(learning_rate)

            if i % percentage == 0:
                ip = i / percentage
                print('\rProgress: [%d%%]' % ip, end="")
                print(" | Loss: " + str(self.cost[-1]), end="")
                self.forward_propagation(X_test)
                current_validation_loss = self.calculate_cost(Y_test)
                print(" , Validation loss: ", current_validation_loss, end="")
                if current_validation_loss >= previous_validation_loss:
                    print(" Increasing. ↑ | ", end="")
                else:
                    print(" Decreasing. ↓ | ", end="")
                previous_validation_loss = current_validation_loss

        print("")
        print(self.cost[::100])

    def evaluate(self, test_data, test_labels):
        self.forward_propagation(test_data)
        predictions = np.round(self.last_activation)

        predictions = np.argmax(predictions, axis=0)
        test_labels = np.argmax(test_labels, axis=0)
        # predictions = np.round(predictions)

        print("Evaluation percentage: ", np.mean(predictions == test_labels) * 100)
        print("Exemplary predictions: ", predictions[:10])
        print("Correct answers:       ", test_labels[:10])

    def save_network(self, name):
        file = open(name, 'wb')
        pkl.dump(self.layers_list, file)
        pkl.dump(self.cost, file)
        pkl.dump(self.cost_function, file)
        file.close()

    def load_network(self, name):
        data = []
        with open(name, "rb") as f:
            while True:
                try:
                    data.append(pkl.load(f))
                except EOFError:
                    break
        self.layers_list = data[0]
        self.cost = data[1]
        self.cost_function = data[2]

    def visualize_weights(self):
        for layer in self.layers_list:
            plt.imshow(layer.get_weights())
            plt.show()
