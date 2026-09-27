"""
Minimal-prep LightGBM baseline for the Ames Housing dataset.
 
Download the CSV from:
https://www.kaggle.com/datasets/prevek18/ames-housing-dataset?resource=download
 
and point CSV_PATH below at it (the file is usually called "AmesHousing.csv").
"""

# ---- 0. Imports ----
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    root_mean_squared_error, 
    mean_absolute_error,
    mean_absolute_percentage_error
)
import lightgbm as lgb
import mlflow
import mlflow.lightgbm

# ---- 1. Load data----
df = pd.read_csv('data/AmesHousing.csv')

# Drop ID columns that are not useful for modeling
df = df.drop(columns = ['PID', 'Order'])

# ---- 2. Prepare data (minimal effort) ----

# Target variable
TARGET = 'SalePrice'

# Apply log transformation to the target variable to reduce skewness
y = np.log1p(df[TARGET])
X = df.drop(columns = [TARGET])

# Convert every non-numeric column to pandas 'category' dtype.
# LightGBM will handle these natively and handles NaNs on its own.
cat_cols = X.select_dtypes(include=["object", "str"]).columns.tolist()
 
# A few columns are stored as integers but are really category codes.
#   - MS SubClass: pure ID, no order
#   - Yr Sold / Mo Sold
numeric_but_categorical = [c for c in ["MS SubClass", "Yr Sold", "Mo Sold"] if c in X.columns]
cat_cols += numeric_but_categorical

for c in cat_cols:
    X[c] = X[c].astype("category")
    
# ---- 3. Split ----

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---- 4. Train ----

mlflow.set_experiment("ames-housing-regression")

with mlflow.start_run():
    mlflow.log_params({"n_estimators": 2000, "learning_rate": 0.03, "num_leaves": 31, "random_state": 42})
    model = lgb.LGBMRegressor(
        n_estimators=2000,
        learning_rate=0.03,
        num_leaves=31,
        random_state=42,
    )
 
    model.fit(
        X_train,
        y_train,
        eval_X = X_test,
        eval_y = y_test,
        eval_metric="rmse",
        categorical_feature=cat_cols,
        callbacks=[lgb.early_stopping(stopping_rounds=100), lgb.log_evaluation(period=100)],
    )

    # ---- 5. Evaluate (on original $ scale) ----

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
    
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("mape", mape)
    mlflow.log_metric("median_ape", median_ape)
    mlflow.lightgbm.log_model(model, "model")

# ---- 6. Feature importance ----

importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nTop 15 features by importance:")
print(importances.head(15))