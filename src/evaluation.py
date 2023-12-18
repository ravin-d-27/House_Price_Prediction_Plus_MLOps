import logging
from abc import ABC, abstractmethod
import numpy as np


class Evaluation(ABC):
    """Abstracts class defining strategy for evaluation of Models"""
    
    def calculate_scores(self, y_true:np.ndarray, y_pred:np.ndarray):
        """
        This method calculates the scores for the model.
        Args:
            y_true: True values
            y_pred: Predicted values
        Returns:
            _type_: Scores for the model
        """
        pass
    
    
class MSE(Evaluation):
    """
    This class implements the strategy for calculating MSE score.
    """
    
    