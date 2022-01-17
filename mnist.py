import numpy as np
import struct

def toOneHot(y):
    X = np.zeros((len(y), 10))
    for i in range(len(y)):
        X[i][y[i]] = 1
    return X

def load_images(file_name):
    binfile = open(file_name, 'rb')
    buffers = binfile.read()
    magic,num,rows,cols = struct.unpack_from('>IIII',buffers, 0)
    bits = num * rows * cols
    images = struct.unpack_from('>' + str(bits) + 'B', buffers, struct.calcsize('>IIII'))
    binfile.close()
    images = np.reshape(images, [num, rows * cols])
    return images

def load_labels(file_name):
    binfile = open(file_name, 'rb')
    buffers = binfile.read()
    magic,num = struct.unpack_from('>II', buffers, 0)
    labels = struct.unpack_from('>' + str(num) + "B", buffers, struct.calcsize('>II'))
    binfile.close()
    labels = np.reshape(labels, [num])
    return labels

def load_datasets():
    train_images = load_images('datasets/train-images.idx3-ubyte')
    train_labels = load_labels('datasets/train-labels.idx1-ubyte')
    test_images = load_images('datasets/t10k-images.idx3-ubyte')
    test_labels = load_labels('datasets/t10k-labels.idx1-ubyte')
    return train_images, toOneHot(train_labels), test_images, toOneHot(test_labels)