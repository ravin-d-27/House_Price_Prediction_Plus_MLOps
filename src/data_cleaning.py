import logging
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Union

class DataStrategy(ABC):
    """
    This is the Abstract Base Class for handling data.
    
    """
    
    @abstractmethod
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        """
        This method handles data.
        
        Args:
            data: raw data
        Returns:
            pd.DataFrame: cleaned data
        """
        pass
    

class DataPreprocessingStrategy(DataStrategy):
    
    def handle_data(self, data: pd.DataFrame)->pd.DataFrame:
        """
        This method handles data.
        
        Args:
            data: raw data
        Returns:
            pd.DataFrame: cleaned data
        """
        
        try:
            data = data.replace({'airport':{'YES':1, 'NO':0}, 'waterbody':{'River':0, 'Lake':1, 'Lake and River':2, 'None':3}})
        except:
            logging.error("Failed to handle data.")
            raise Exception("Failed to handle data.")