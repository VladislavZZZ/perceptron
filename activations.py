import numpy as np


def sigmoid(x):
    return 1. / (1. + np.exp(-x))


def prime_sigmoid(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z) * (1 - sigmoid(z))
