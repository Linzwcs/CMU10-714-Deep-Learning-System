from typing import List, Optional
from ..data_basic import Dataset
import numpy as np
import gzip


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

    data_size = int.from_bytes(bytes_images[4:8])
    image_H = int.from_bytes(bytes_images[8:12])
    image_W = int.from_bytes(bytes_images[12:16])
    bytes_lists = [pix for pix in bytes_images[16:]]
    images = np.array(bytes_lists, dtype=np.float32).reshape(-1, image_H * image_W)
    images = images / 255
    assert images.shape == (data_size, image_H * image_W)

    with open(label_filename, "rb") as f:
        bytes_labels = gzip.decompress(f.read())

    data_size = int.from_bytes(bytes_labels[4:8])
    labels = [i for i in bytes_labels[8:]]
    labels = np.array(labels, dtype=np.uint8)
    assert labels.shape == (data_size,)

    return images, labels


class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        ### BEGIN YOUR SOLUTION
        super().__init__(transforms)
        self.images, self.labels = parse_mnist(image_filename, label_filename)
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        ### BEGIN YOUR SOLUTION
        imgs, labels = self.images[index], self.labels[index]
        # print(len(imgs.shape))
        if len(imgs.shape) > 1:
            imgs = np.vstack(
                [[self.apply_transforms(img.reshape(28, 28, 1)) for img in imgs]]
            )
        else:
            imgs = self.apply_transforms(imgs.reshape(28, 28, 1))
        # print(index, imgs.shape)
        return (imgs, labels)
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        ### BEGIN YOUR SOLUTION
        return len(self.images)
        ### END YOUR SOLUTION
