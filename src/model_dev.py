import logging
from abc import ABC, abstractmethod
from sklearn.svm import SVR

class Model(ABC):
    
    @abstractmethod
    def train(self, X_train, y_train):
        
        """
        This method trains the model.
        Args:
        X_train: Training data
        y_train: Target data
        """
        pass
    
class SupportVectorMachine(Model):
    
    def train(self, X_train, y_train):
        """

        Args:
            X_train (_type_): Training data
            y_train (_type_): Target data
        Returns:
            _type_: Trained model
        """
        try:
            model = SVR(kernel='linear')
            model.fit(X_train, y_train)
            logging.info("Model trained successfully.")
            return model
        except Exception as e:
            logging.error("Error in training the model: {}".format(e))
            raise e