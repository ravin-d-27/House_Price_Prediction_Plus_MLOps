import logging
from abc import ABC, abstractmethod
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

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
    
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("Calculating MSE score")
            mse = mean_squared_error(y_true, y_pred)
            logging.info("MSE score: {}".format(mse))
            return mse
        except Exception as e:
            logging.error("Error in calculating MSE score: {}".format(e))
            raise e
        
class RMSE(Evaluation):
    """
    This class implements the strategy for calculating RMSE score.
    """
    
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("Calculating RMSE score")
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            logging.info("RMSE score: {}".format(rmse))
            return rmse
        except Exception as e:
            logging.error("Error in calculating RMSE score: {}".format(e))
            raise e
        
class R2(Evaluation):
    """
    This class implements the strategy for calculating RMSE score.
    """
    
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("Calculating RMSE score")
            r2 = r2_score(y_true, y_pred)
            logging.info("R2 score: {}".format(r2))
            return r2
        except Exception as e:
            logging.error("Error in calculating R2 score: {}".format(e))
            raise e
    