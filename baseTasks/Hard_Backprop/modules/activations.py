import numpy as np
from .base import Module
from scipy.special import expit, softmax


class ReLU(Module):
    """
    Applies element-wise ReLU function
    """
    def compute_output(self, input: np.ndarray) -> np.ndarray:
        """
        :param input: array of an arbitrary size
        :return: array of the same size
        """
        return np.maximum(input, 0)

    def compute_grad_input(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        """
        :param input: array of an arbitrary size
        :param grad_output: array of the same size
        :return: array of the same size
        """
        return grad_output * (input > 0)


class Sigmoid(Module):
    """
    Applies element-wise sigmoid function
    """
    def compute_output(self, input: np.ndarray) -> np.ndarray:
        """
        :param input: array of an arbitrary size
        :return: array of the same size
        """
        # 1 / (1 + np.e ** -input)
        return expit(input)

    def compute_grad_input(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        """
        :param input: array of an arbitrary size
        :param grad_output: array of the same size
        :return: array of the same size
        """
        # 1 / (1 + np.e ** -input)
        sig = expit(input)
        return grad_output * sig * (1 - sig)


class Softmax(Module):
    """
    Applies Softmax operator over the last dimension
    """
    def compute_output(self, input: np.ndarray) -> np.ndarray:
        """
        :param input: array of size (batch_size, num_classes)
        :return: array of the same size
        """
        # e_input = np.e ** (input - np.max(input, axis=1, keepdims=True))
        # e_sum = np.sum(e_input, axis=1, keepdims=True)
        # e_input / e_sum
        return softmax(input, axis=-1)

    def compute_grad_input(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        """
        :param input: array of size (batch_size, num_classes)
        :param grad_output: array of the same size
        :return: array of the same size
        """
        probs = softmax(input, axis=-1)
        sum_grad = np.sum(grad_output * probs, axis=1, keepdims=True)
        return probs * (grad_output - sum_grad)


class LogSoftmax(Module):
    """
    Applies LogSoftmax operator over the last dimension
    """
    def compute_output(self, input: np.ndarray) -> np.ndarray:
        """
        :param input: array of size (batch_size, num_classes)
        :return: array of the same size
        """
        return np.log(softmax(input, axis=-1))

    def compute_grad_input(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        """
        :param input: array of size (batch_size, num_classes)
        :param grad_output: array of the same size
        :return: array of the same size
        """
        probs = softmax(input, axis=-1)
        sum_grad = np.sum(grad_output, axis=1, keepdims=True)
        return grad_output - probs * sum_grad
