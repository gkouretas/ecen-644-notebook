import abc
import numpy as np

class ActivationFunction(abc.ABC):
    """Abstract class for activation function container"""
    @abc.abstractmethod
    def compute(x: np.ndarray) -> np.ndarray:
        pass

    @abc.abstractmethod
    def derivative(x: np.ndarray) -> np.ndarray:
        pass

class ReLU(ActivationFunction):
    """ReLU activation function"""
    @staticmethod
    def compute(x: np.ndarray): 
        return np.maximum(x, 0.0)
    
    @staticmethod
    def derivative(x: np.ndarray):
        return np.array(x > 0.0) * 1.0 

class Sigmoid(ActivationFunction):
    """Sigmoid activation function"""
    @staticmethod
    def compute(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-x))
    
    @staticmethod
    def derivative(x: np.ndarray) -> np.ndarray:
        return x * (1.0 - x)

class TanH(ActivationFunction):
    """tanh activation function"""
    @staticmethod
    def compute(x: np.ndarray) -> np.ndarray:
        return np.tanh(x)

    @staticmethod
    def derivative(x: np.ndarray) -> np.ndarray:
        return 1 - np.power(np.tanh(x), 2)
