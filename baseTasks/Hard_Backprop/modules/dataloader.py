import numpy as np
from math import ceil


class DataLoader(object):
    """
    Tool for shuffling data and forming mini-batches
    """
    def __init__(self, X, y, batch_size=1, shuffle=False):
        """
        :param X: dataset features
        :param y: dataset targets
        :param batch_size: size of mini-batch to form
        :param shuffle: whether to shuffle dataset
        """
        assert X.shape[0] == y.shape[0]
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.batch_id = 0  # use in __next__, reset in __iter__

    def __len__(self) -> int:
        """
        :return: number of batches per epoch
        """
        return ceil(self.X.shape[0] / self.batch_size)

    def num_samples(self) -> int:
        """
        :return: number of data samples
        """
        return self.X.shape[0]

    def __iter__(self):
        """
        Shuffle data samples if required
        :return: self
        """
        self.batch_id = 0
        if self.shuffle:
            indices = np.random.permutation(self.X.shape[0])
            self.X = self.X[indices]
            self.y = self.y[indices]
        return self

    def __next__(self):
        """
        Form and return next data batch
        :return: (x_batch, y_batch)
        """
        if self.batch_id >= len(self):
            raise StopIteration

        curr_range = self.batch_id * self.batch_size
        self.batch_id += 1
        return (self.X[curr_range: curr_range + self.batch_size], self.y[curr_range: curr_range + self.batch_size])
