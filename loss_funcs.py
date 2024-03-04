import numpy as np
import abc

class LossFunction:
    @abc.abstractmethod
    def compute(actual: np.ndarray, predicted: np.ndarray): pass

    @abc.abstractmethod
    def derivative(actual: np.ndarray, predicted: np.ndarray): pass

class BinaryXEntropyLoss(LossFunction):
    @staticmethod
    def compute(actual: np.ndarray, predicted: np.ndarray):
        return -np.average(
            actual*np.log(predicted) + (1.0-actual)*np.log(1 - predicted)
        )

    @staticmethod
    def derivative(actual: np.ndarray, predicted: np.ndarray):
        return 1/np.size(actual, axis = 0) * ((1.0-actual)/(1.0-predicted) - predicted/actual) 