import logging
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from typing import Union
from sklearn.calibration import column_or_1d

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
        """
        Removes columns which are not required, fills missing values with median average values, and converts the data type to float.
        """
        try:
            data = data.drop(
                [
                    "order_approved_at",
                    "order_delivered_carrier_date",
                    "order_delivered_customer_date",
                    "order_estimated_delivery_date",
                    "order_purchase_timestamp",
                ],
                axis=1,
            )
            data["product_weight_g"].fillna(data["product_weight_g"].median(), inplace=True)
            data["product_length_cm"].fillna(data["product_length_cm"].median(), inplace=True)
            data["product_height_cm"].fillna(data["product_height_cm"].median(), inplace=True)
            data["product_width_cm"].fillna(data["product_width_cm"].median(), inplace=True)
            # write "No review" in review_comment_message column

            data = data.select_dtypes(include=[np.number])
            cols_to_drop = ["customer_zip_code_prefix", "order_item_id", "review_comment_message"]
            data = data.drop(cols_to_drop, axis=1)

            return data
        except Exception as e:
            logging.error(e)
            raise e
        
class DataDivideStrategy(DataStrategy):
    
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        try:
            X = data.drop("review_score", axis=1)
            y = data["review_score"].ravel()
            y = y.ravel()  # Use ravel() to ensure y is a 1D array
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
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
    print(data_cleaning.data.head())
    print("Done!")        