import network as net
import denseLayer
import activations as act
import costFunctions as cst
import numpy as np
import sklearn as skl
import os
from scipy.io import loadmat


def parse_400():
    data = loadmat(os.path.join('Data', 'C:/Users/jacpy/PycharmProjects/NumpyNN/ex4data1.mat'))
    X, y = data['X'], data['y'].ravel()
    np.random.seed(4321)
    X, y = skl.utils.shuffle(X, y)

    y[y == 10] = 0
    Y = y.reshape(-1)
    Y = np.eye(10)[Y]

    print(Y.shape)
    return X, Y, y


def func():
    model = net.Model(cst.CrossEntropy)

    model.sequential([
        denseLayer.DenseLayer(400, 100, act.ReLU),
        denseLayer.DenseLayer(100, 25, act.ReLU),
        denseLayer.DenseLayer(25, 10, act.Sigmoid),
    ])

    X, Y, y = parse_400()
    X, Y = X.T, Y.T

    print(X.shape, Y.shape, y.shape)

    model.train(X, Y, .3, 1000)
    model.evaluate(X, Y)


if __name__ == '__main__':
    func()
