"""
import os

from tensorflow.examples.tutorials.mnist import input_data

inputs_dir = os.getenv('VH_INPUTS_DIR', 'data')
data_set_files = [
    os.path.join(inputs_dir, 'training-set-images/train-images-idx3-ubyte.gz'),
    os.path.join(inputs_dir, 'training-set-labels/train-labels-idx1-ubyte.gz'),
    os.path.join(inputs_dir, 'test-set-images/t10k-images-idx3-ubyte.gz'),
    os.path.join(inputs_dir, 'test-set-labels/t10k-labels-idx1-ubyte.gz'),
]


mnist = input_data.read_data_sets(inputs_dir, one_hot=True)
print(mnist.train)
"""

from keras.datasets import mnist
from keras.utils import np_utils


def load_mnist():
    mnist.load_data()
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.reshape(60000, 28, 28, 1)
    X_test = X_test.reshape(10000, 28, 28, 1)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    X_train /= 255.0
    X_test /= 255.0
    y_train = np_utils.to_categorical(y_train, 10)
    y_test = np_utils.to_categorical(y_test, 10)

    return (X_train, y_train), (X_test, y_test)
