import abc
import numpy as np

class ActivationFunction(abc.ABC):
    """Abstract class for activation function container"""
    def __init__(self) -> None:
        pass

    @abc.abstractmethod
    def compute(self, x: np.ndarray) -> np.ndarray:
        pass

    @abc.abstractmethod
    def derivative(self, x: np.ndarray) -> np.ndarray:
        pass

class ReLU(ActivationFunction):
    """ReLU activation function"""
    def compute(self, x: np.ndarray): 
        return np.maximum(x, 0.0)
    
    def derivative(self, x: np.ndarray):
        return np.array(x > 0.0) * 1.0 

class Sigmoid(ActivationFunction):
    """Sigmoid activation function"""
    def __init__(self) -> None:
        super().__init__()

    def compute(self, x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-x))
    
    def derivative(self, x: np.ndarray) -> np.ndarray:
        return x * (1.0 - x)

class TanH(ActivationFunction):
    """tanh activation function"""
    def __init__(self) -> None:
        super().__init__()

    def compute(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(x)

    def derivative(self, x: np.ndarray) -> np.ndarray:
        return 1 - np.power(np.tanh(x), 2)
