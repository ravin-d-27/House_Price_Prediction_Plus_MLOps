from zenml.steps import BaseParameters

class ModelNameConfig(Base):
    """Model Configs"""
    
    model_name: str = "SupportVectorMachine"