from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import sklearn as skl
import os
from scipy.io import loadmat
from mlxtend.data import loadlocal_mnist
import platform

size = 64


def show_examples():
    number = np.random.randint(0, 1000)
    image = Image.open("C:/Users/jacpy/PycharmProjects/NumpyNN/PetImages/Cat/" + str(number) + ".jpg")

    image = image.resize((size, size), Image.ANTIALIAS)

    print(image.size)

    data = np.asarray(image)
    print(type(data))
    # summarize shape
    print(data.shape)

    # create Pillow image
    image2 = Image.fromarray(data)
    print(type(image2))

    # summarize image details
    print(image2.mode)
    print(image2.size)

    plt.imshow(data[:, :, 0])
    plt.show()


def crawl(folder_images, max_images, y_value):
    x_data = np.zeros((size * size * 3))
    y_data = np.full((max_images, 1), y_value)

    folder_images = "C:/Users/jacpy/PycharmProjects/NumpyNN/PetImages/" + folder_images

    percentage = max_images / 100

    for _, _, image_filenames in os.walk(folder_images):
        for image_filename in image_filenames:

            image_filename = folder_images + image_filename
            try:

                if x_data.shape[0] % percentage == 0:
                    ip = x_data.shape[0] / percentage
                    print('\rProgress: [%d%%]' % ip, end="")

                img = Image.open(image_filename)
                img = img.resize((size, size), Image.ANTIALIAS)
                data = np.asarray(img)
                data = data.flatten()
                x_data = np.vstack([x_data, data])
                if x_data.shape[0] == max_images:
                    print("\nFolder Done. ")
                    return x_data, y_data
                if x_data.shape[0] == max_images:
                    raise Exception("Internal Error")
            except IndexError:
                print("\n Too many indices for array: array is 2-dimensional, but 3 were indexed")
            except ValueError:
                print("\n Too many indices for array: array is 2-dimensional, but 3 were indexed")
    print("Folder Done. ")


def get_data(max_images):
    x_cat, y_cat = crawl("Cat/", max_images, 1)
    x_dog, y_dog = crawl("Dog/", max_images, 0)
    X, Y = np.concatenate((x_dog, x_cat), axis=0), np.concatenate((y_dog, y_cat), axis=0)
    X = X / 256
    np.random.seed(43212)
    X, Y = skl.utils.shuffle(X, Y)
    return X, Y


def parse_400():
    data = loadmat(os.path.join('Data', 'C:/Users/jacpy/PycharmProjects/NumpyNN/ex4data1.mat'))
    X, y = data['X'], data['y'].ravel()
    np.random.seed(5321)
    X, y = skl.utils.shuffle(X, y)

    y[y == 10] = 0
    Y = y.reshape(-1)
    Y = np.eye(10)[Y]

    print(Y.shape)
    return X, Y, y

def parse_784():
    if not platform.system() == 'Windows':
        X, y = loadlocal_mnist(
            images_path='C:/Users/jacpy/PycharmProjects/NumpyNN/mnist/t10k-images-idx3-ubyte',
            labels_path='C:/Users/jacpy/PycharmProjects/NumpyNN/mnist/train-labels-idx1-ubyte')

    else:
        X, y = loadlocal_mnist(
            images_path='C:/Users/jacpy/PycharmProjects/NumpyNN/mnist/train-images.idx3-ubyte',
            labels_path='C:/Users/jacpy/PycharmProjects/NumpyNN/mnist/train-labels.idx1-ubyte')
    X, y = skl.utils.shuffle(X, y)

    print("X: ", X.shape)
    X = X / 255

    Y = y.reshape(-1)
    Y = np.eye(10)[Y]
    print("Y: ", Y.shape)
    return X, Y, y
