from zenml import pipeline
from steps.ingest_data import ingest_data
from steps.clean_data import clean_data
from steps.model_train import train_model
from steps.evaluation import evaluate_model

@pipeline
def training_pipeline(data_path: str):
    """
    Training pipeline.
    """
    df = ingest_data(data_path)
    print("\n\n\nData Loaded\n\n\n")
    X_train, X_test, y_train, y_test = clean_data(df)
    print("\n\n\nData Cleaned\n\n\n")
    model = train_model(X_train, X_test, y_train, y_test)
    print("\n\n\Model Trained\n\n\n")
    mse, rmse, r2 = evaluate_model(model, X_test, y_test)