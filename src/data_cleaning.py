import logging
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from typing import Union

class DataStrategy(ABC):
    """
    This is the Abstract Base Class for handling data.
    
    """
    
    @abstractmethod
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        pass
    

class DataPreprocessingStrategy(DataStrategy):
    
    def handle_data(self, data: pd.DataFrame)->pd.DataFrame:
        try:
            data = data.replace({'airport':{'YES':1, 'NO':0}, 'waterbody':{'River':0, 'Lake':1, 'Lake and River':2, 'None':3}})
            
            data.replace({"airport": {"YES": 1, "NO": 0}, 
                           "waterbody": {"River": 1, "Lake": 2, "Lake and River": 3, "None": 4}
                           ,}, inplace=True)

            data = data.drop(['bus_ter'],axis=1)
            
        except:
            logging.error("Failed to handle data.")
            raise Exception("Failed to handle data.")