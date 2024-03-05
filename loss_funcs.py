import numpy as np
import abc

class LossFunction:
    @abc.abstractmethod
    def compute(actual: np.ndarray, predicted: np.ndarray): pass

    @abc.abstractmethod
    def derivative(actual: np.ndarray, predicted: np.ndarray): pass

class MSE(LossFunction):
    @staticmethod
    def compute(actual: np.ndarray, predicted: np.ndarray):
        return np.sum(np.power(predicted - actual, 2))

    @staticmethod
    def derivative(actual: np.ndarray, predicted: np.ndarray):
        return 2 * (predicted - actual) / np.size(actual)