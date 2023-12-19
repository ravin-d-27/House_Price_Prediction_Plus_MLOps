import logging
from zenml import step
import pandas as pd

from src.evaluation import Evaluation, MSE, RMSE, R2
from sklearn.base import RegressorMixin

from typing import Tuple
from typing_extensions import Annotated

@step
def evaluate_model(model: RegressorMixin, 
                   X_test: pd.DataFrame,
                   y_test: pd.DataFrame)-> Tuple[
                       Annotated[float,'MSE'],
                       Annotated[float,'RMSE'],
                       Annotated[float,'R2'],
                       ]:
    """
    Evaluate model.
    
    Args:
        df: cleaned data
    Returns:
        pd.DataFrame: evaluation results
    """
    
    try:
        predictions = model.predict(X_test)
        mse_class = MSE()
        mse = mse_class.calculate_scores(y_test, predictions)
        
        
        rmse_class = RMSE()
        rmse = rmse_class.calculate_scores(y_test, predictions)
        
        
        r2_class = R2()
        r2 = r2_class.calculate_scores(y_test, predictions)
        
        return mse, rmse, r2
    
    except Exception as e:
        logging.error("Error in evaluating model: {}".format(e))
        raise e