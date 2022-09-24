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


def cross_entropy_loss(actual, target):
    loss = -np.sum(actual * np.log(target))
    return loss / float(actual.shape[0])

