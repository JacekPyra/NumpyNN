import network as net
from layers import denseLayer, dropout
import activations as af
import costFunctions as cst

import data_parser




def func():
    model = net.Model(cst.CrossEntropy)

    model.sequential([
        denseLayer.DenseLayer(400, 100, af.Tanh),  # standard deviation = sqrt(1 / input features)
        denseLayer.DenseLayer(100, 50, af.Tanh),
        dropout.Dropout(0.9),
        denseLayer.DenseLayer(50, 20, af.Tanh),
        dropout.Dropout(0.9),
        denseLayer.DenseLayer(20, 10, af.Sigmoid),
    ])
    #model.load_network("Weights.pkl")
    X, Y, y = data_parser.parse_400()
    X, Y = X.T, Y.T
    pivot = 4000
    X_train, X_test = X[:, :pivot], X[:, pivot:]
    Y_train, Y_test = Y[:, :pivot], Y[:, pivot:]

    model.train(0.05, 1000, X_train, Y_train, X_test, Y_test)
    model.evaluate(X_test, Y_test)
    model.save_network("Weights.pkl")


if __name__ == '__main__':
    func()
