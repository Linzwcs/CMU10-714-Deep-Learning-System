"""hw1/apps/simple_ml.py"""

import struct
import gzip
import numpy as np

import sys

sys.path.append("python/")
import needle as ndl


def parse_mnist(image_filesname, label_filename):
    """Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded
                data.  The dimensionality of the data should be
                (num_examples x input_dim) where 'input_dim' is the full
                dimension of the data, e.g., since MNIST images are 28x28, it
                will be 784.  Values should be of type np.float32, and the data
                should be normalized to have a minimum value of 0.0 and a
                maximum value of 1.0.

            y (numpy.ndarray[dypte=np.int8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.int8 and
                for MNIST will contain the values 0-9.
    """
    ### BEGIN YOUR SOLUTION
    with open(image_filesname, "rb") as f:
        bytes_images = gzip.decompress(f.read())
    # magic_num = int.from_bytes(bytes_images[:4])
    data_size = int.from_bytes(bytes_images[4:8])
    image_H = int.from_bytes(bytes_images[8:12])
    image_W = int.from_bytes(bytes_images[12:16])
    bytes_lists = [pix for pix in bytes_images[16:]]
    images = np.array(bytes_lists, dtype=np.float32).reshape(-1, image_H * image_W)
    images = images / 255
    assert images.shape == (data_size, image_H * image_W)

    with open(label_filename, "rb") as f:
        bytes_labels = gzip.decompress(f.read())
    # magic_num = int.from_bytes(bytes_images[:4])
    data_size = int.from_bytes(bytes_labels[4:8])
    labels = [i for i in bytes_labels[8:]]
    labels = np.array(labels, dtype=np.uint8)
    assert labels.shape == (data_size,)

    return images, labels
    ### END YOUR SOLUTION


def softmax_loss(Z, y_one_hot):
    """Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (ndl.Tensor[np.float32]): 2D Tensor of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (ndl.Tensor[np.int8]): 2D Tensor of shape (batch_size, num_classes)
            containing a 1 at the index of the true label of each example and
            zeros elsewhere.

    Returns:
        Average softmax loss over the sample. (ndl.Tensor[np.float32])
    """
    ### BEGIN YOUR SOLUTION
    bsz = Z.shape[0]
    return (ndl.log(ndl.exp(Z).sum((1,))) - (y_one_hot * Z).sum((1,))).sum() / bsz
    ### END YOUR SOLUTION


def nn_epoch(X, y, W1, W2, lr=0.1, batch=100):
    """Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X * W1) * W1
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        W1 (ndl.Tensor[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)
        W2 (ndl.Tensor[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD mini-batch

    Returns:
        Tuple: (W1, W2)
            W1: ndl.Tensor[np.float32]
            W2: ndl.Tensor[np.float32]
    """

    ### BEGIN YOUR SOLUTION
    num_class = W2.shape[1]
    onehot = np.eye(num_class)
    assert X.shape[0] == y.shape[0]
    # print(X.shape, y.shape, batch)

    for idx in range(0, len(X), batch):

        W1 = ndl.Tensor(W1)
        W2 = ndl.Tensor(W2)
        batch_X = X[idx : idx + batch]
        batch_y = y[idx : idx + batch]
        batch_Iy = onehot[batch_y]

        batch_X = ndl.Tensor(batch_X)
        batch_Iy = ndl.Tensor(batch_Iy)

        loss = softmax_loss(ndl.relu(batch_X @ W1) @ W2, batch_Iy)
        loss.backward()

        W1, W1_grad = W1.realize_cached_data(), W1.grad.realize_cached_data()
        W2, W2_grad = W2.realize_cached_data(), W2.grad.realize_cached_data()
        W1 -= W1_grad * lr
        W2 -= W2_grad * lr
    return ndl.Tensor(W1), ndl.Tensor(W2)
    ### END YOUR SOLUTION


### CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT


def loss_err(h, y):
    """Helper function to compute both loss and error"""
    y_one_hot = np.zeros((y.shape[0], h.shape[-1]))
    y_one_hot[np.arange(y.size), y] = 1
    y_ = ndl.Tensor(y_one_hot)
    return softmax_loss(h, y_).numpy(), np.mean(h.numpy().argmax(axis=1) != y)
