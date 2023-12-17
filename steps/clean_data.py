import logging
import pandas as pd
from zenml import step

from src.data_cleaning import DataCleaning, DataDivideStrategy, DataPreprocessingStrategy

from typing_extensions import Annotated
from typing import Tuple

@step
def clean_data(df:pd.DataFrame)->Tuple(
    Annotated[pd.DataFrame, "X_train"],
    Annotated[pd.DataFrame, "X_test"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.Series, "y_test"]
    
):
    """
    Clean data.
    
    Args:
        df: raw data
    Returns:
        pd.DataFrame: cleaned data
    """
    try:
        process_strategy = DataPreprocessingStrategy()
        data_cleaning = DataCleaning()
        processed_data = data_cleaning.handle_data()
        
        divide_strategy = DataDivideStrategy()
        data_cleaning = DataCleaning(processed_data, divide_strategy)
        X_train, X_test, y_train, y_test = data_cleaning.handle_data()
        
        logging.info("Data Cleaning Completed")
    except Exception as e:
        logging.error("Error in Data Cleaning: {}".format(e))
        raise e