import logging

import pandas as pd
from zenml import step

class IngestData:
    """
    Ingest data from a given path.
    """
    def __init__(self, data_path:str):
        
        """
        Args:
            data_path: path to the data
        """
        self.data_path = data_path
        
    def get_data(self):
        """
        Ingest data from a given path.
        """
        logging.info(f"Ingesting data from {self.data_path}")
        dataframe = pd.read_csv(self.data_path)
        print(dataframe.head())
        return dataframe
    
@step
def ingest_data(data_path: str) -> pd.DataFrame:
    """
    Ingesting data from a given path.
    
    Args:
        data_path: path to the data
    Returns:
        pd.DataFrame: data
    """
    try:
        print("Before calling get_data")
        ingest = IngestData(data_path)
        df = ingest.get_data()  # Call the get_data method to get the DataFrame
        logging.info("Ingesting data completed")
        return df
    except Exception as e:
        logging.error(f"Failed to ingest data from {data_path}")
        raise e

    
