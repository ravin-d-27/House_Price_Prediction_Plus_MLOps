import logging
import pandas as pd
from zenml import step

@step
def clean_data(df:pd.DataFrame)->None:
    """
    Clean data.
    
    Args:
        df: raw data
    Returns:
        pd.DataFrame: cleaned data
    """
    pass