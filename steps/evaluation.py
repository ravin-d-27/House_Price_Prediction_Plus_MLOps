import logging
from zenml import step
import pandas as pd

@step
def evaluate_model(df:pd.DataFrame)->None:
    """
    Evaluate model.
    
    Args:
        df: cleaned data
    Returns:
        pd.DataFrame: evaluation results
    """
    pass