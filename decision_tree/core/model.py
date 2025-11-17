from abc import ABC , abstractmethod

class FitPrediction(ABC):
    
    @abstractmethod
    def fit(self, X,y):...
    
    @abstractmethod
    def predict(self, X):...
