import logging
import pandas as pd
from zenml import step

from src.model_dev import SupportVectorMachine
from sklearn.base import RegressorMixin
from .config import ModelNameConfig

@step
def train_model(X_train:pd.DataFrame,
                X_test:pd.DataFrame,
                y_train:pd.DataFrame,
                y_test:pd.DataFrame,
                config: ModelNameConfig)->RegressorMixin:
    """
    Train model.
    
    Args:
        df: cleaned data
    Returns:
        pd.DataFrame: trained model
    """
    
    try:
        model = None
        if config.model_name == "SupportVectorMachine":
            model = SupportVectorMachine()
            trained_model = model.train(X_train, y_train)
            return trained_model
        else:
            raise ValueError("Model {} not found in config.".format(config.model_name))
    except Exception as e:
        logging.error("Error in training the model: {}".format(e))
        raise e