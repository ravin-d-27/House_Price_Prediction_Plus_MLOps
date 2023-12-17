import logging
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from typing import Union

from sklearn.model_selection import train_test_split

from pandas.core.api import Series as Series

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
        
class DataDivideStrategy(DataStrategy):
    
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        
        try:
            X = data.iloc[:,1:].values
            y = data.iloc[:,0].values
            
            X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=1234)
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logging.error("Error in spliting the Data: {}".format(e))
            raise e
        

class DataCleaning():
    
    def __init__(self,data:pd.DataFrame, strategy:DataStrategy) -> None:
        self.data = data
        self.strategy = strategy
        
    def handle_data(self)->Union[pd.DataFrame, pd.Series]:
        try:
            return self.strategy.handle_data(self.data)
        except Exception as e:
            logging.error("Error in Handling the Data")
            raise e

if __name__ == "__main__":
    data = pd.read_csv('/home/ravind27/Desktop/My_Projects_and_Codes/House_Price_Prediction_Plus_MLOps/data/House_Price.csv')
    data_cleaning = DataCleaning(data, DataPreprocessingStrategy())
    data_cleaning.handle_data()
    print("Done!")        