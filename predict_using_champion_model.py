import mlflow

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    root_mean_squared_error, 
    mean_absolute_error,
    mean_absolute_percentage_error
)
import mlflow
import mlflow.lightgbm
from src.prepare_input import prepare_X

# ---- Configuration ----
from config import MODEL_NAME, TARGET

# ---- 1. Load data----
df = pd.read_csv('data/AmesHousing.csv')

# Drop ID columns that are not useful for modeling
df = df.drop(columns = ['PID', 'Order'])

# ---- 2. Prepare data (minimal effort) ----

# Apply log transformation to the target variable to reduce skewness
y = np.log1p(df[TARGET])
X = df.drop(columns = [TARGET])
X, _ = prepare_X(X)

# ---- 3. Split ----

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = mlflow.lightgbm.load_model(f"models:/{MODEL_NAME}@champion")
pred_log = model.predict(X_test, num_iteration=model.best_iteration_)
pred = np.expm1(pred_log)
y_test_dollars = np.expm1(y_test)

rmse = root_mean_squared_error(y_test_dollars, pred)
mae = mean_absolute_error(y_test_dollars, pred)
mape = mean_absolute_percentage_error(y_test_dollars, pred)
ape = np.abs((y_test_dollars - pred) / y_test_dollars)
median_ape = np.median(ape)

print(f"\nTest RMSE:        ${rmse:,.0f}")
print(f"Test MAE:         ${mae:,.0f}")
print(f"Test MAPE:        {mape:.2%}")
print(f"Test Median APE:  {median_ape:.2%}")