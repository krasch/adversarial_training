import os
"""
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
mnist.load_data()

(X_train, y_train), (X_test, y_test) = mnist.load_data()
print(X_train.shape)