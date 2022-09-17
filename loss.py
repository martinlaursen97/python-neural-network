import numpy as np
import math


def mean_squared(actual, target):
    return (target - actual) ** 2


def difference(actual, target):
    return target - actual


def softmax(x):
    err_sum = np.sum(np.exp(x))
    distribution = np.exp(x) / err_sum
    return distribution


def cross_entropy_loss(x, target):
    desired_idx = np.argmax(target)
    return -math.log(x[0][desired_idx], 2)
