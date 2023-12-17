import logging
import pandas as pd
from zenml import step

@step
def train_model(df:pd.DataFrame)->None:
    """
    Train model.
    
    Args:
        df: cleaned data
    Returns:
        pd.DataFrame: trained model
    """
    pass