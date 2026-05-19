import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import random
import numpy as np

mlflow.set_tracking_uri("http://127.0.0.1:5000/")

mlflow.set_experiment("Used Car Price Prediction")

data = pd.read_csv("used_car_price_dataset_cleaned_scaled.csv")
 
X_train, X_test, y_train, y_test = train_test_split(
    data.drop("selling_price", axis=1),
    data["selling_price"],
    random_state=42,
    test_size=0.2
)

input_example = X_train[0:5]

with mlflow.start_run():
    mlflow.autolog()
    # Train model
    model = RandomForestRegressor()
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        input_example=input_example
    )
    model.fit(X_train, y_train)
    # Log metrics
    accuracy = model.score(X_test, y_test)
    mlflow.log_metric("accuracy", accuracy)

