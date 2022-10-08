import numpy as np
import math


def mean_squared(actual, target):
    return (target - actual) ** 2


def difference(actual, target):
    return target - actual


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def categorical_crossentropy(actual, target):
    return -np.sum(actual * np.log(target + 10**-100))

