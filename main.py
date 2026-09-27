"""
Minimal-prep LightGBM baseline for the Ames Housing dataset.
 
Download the CSV from:
https://www.kaggle.com/datasets/prevek18/ames-housing-dataset?resource=download
 
and point CSV_PATH below at it (the file is usually called "AmesHousing.csv").
"""

# ---- Imports ----

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
import hashlib

# ---- Configuration ----

# Model name
MODEL_NAME = "ames-housing-lgbm"

# Target variable
TARGET = 'SalePrice'

# Hyperparameters for LightGBM
N_ESTIMATORS = 2000
NUM_LEAVES = 20
LEARNING_RATE = .03
RANDOM_STATE = 42

def train_and_log():
    # ---- 1. Load data----
    df = pd.read_csv('data/AmesHousing.csv')

    # Drop ID columns that are not useful for modeling
    df = df.drop(columns = ['PID', 'Order'])

    # ---- 2. Prepare data (minimal effort) ----

    # Apply log transformation to the target variable to reduce skewness
    y = np.log1p(df[TARGET])
    X = df.drop(columns = [TARGET])

    if {"Yr Sold", "Year Built"}.issubset(X.columns):
        # Have a Garage Yr Blt of 2207 in dataset. We guard generically: 
        # a garage "built" after the sale  year is impossible, so treat 
        # it as missing rather than compute a nonsensical negative age.
        if "Garage Yr Blt" in X.columns:
            wrong_garage_built_year = X["Garage Yr Blt"] > X["Yr Sold"]
            X.loc[wrong_garage_built_year, "Garage Yr Blt"] = np.nan
    
        X["house_age_at_sale"] = X["Yr Sold"].astype(float) - X["Year Built"]
        X["is_remodeled"] = (X["Year Remod/Add"] != X["Year Built"]).astype(int)
        X["remodel_age_at_sale"] = X["Yr Sold"].astype(float) - X["Year Remod/Add"]
        if "Garage Yr Blt" in X.columns:
            X["garage_age_at_sale"] = X["Yr Sold"].astype(float) - X["Garage Yr Blt"]
    
        # Drop the absolute-year versions now that we have age-based equivalents
        drop_years = [c for c in ["Year Built", "Year Remod/Add", "Garage Yr Blt"] if c in X.columns]
        X = X.drop(columns=drop_years)

    # Convert every non-numeric column to pandas 'category' dtype.
    # LightGBM will handle these natively and handles NaNs on its own.
    cat_cols = X.select_dtypes(include=["object", "str"]).columns.tolist()
    
    # A few columns are stored as integers but are really category codes.
    #   - MS SubClass: pure ID, no order
    #   - Yr Sold / Mo Sold
    numeric_but_categorical = [c for c in ["MS SubClass", "Yr Sold", "Mo Sold"] if c in X.columns]
    cat_cols += numeric_but_categorical

    for c in cat_cols:
        X[c] = X[c].astype("category").cat.codes
        
    # ---- 3. Split ----

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ---- 4. Train ----

    mlflow.set_experiment("ames-housing-regression")

    with mlflow.start_run() as run:
        feature_hash = hashlib.md5(",".join(sorted(X_train.columns)).encode()).hexdigest()[:8]
        
        mlflow.log_dict(
            {"features": list(X_train.columns)},
            "features.json"
        )
        
        mlflow.log_params({
            "n_estimators": N_ESTIMATORS, 
            "learning_rate": LEARNING_RATE, 
            "num_leaves": NUM_LEAVES, 
            "random_state": RANDOM_STATE,
            "n_features": X_train.shape[1],
            "feature_hash": feature_hash,
        })
        
        model = lgb.LGBMRegressor(
            n_estimators=N_ESTIMATORS,
            learning_rate=LEARNING_RATE,
            num_leaves=NUM_LEAVES,
            random_state=RANDOM_STATE,
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
        model_info = mlflow.lightgbm.log_model(
            model,
            "model",
            input_example=X_train.head(5),
            registered_model_name=MODEL_NAME
        )
        
        run_id = run.info.run_id
        version = model_info.registered_model_version

    # ---- 6. Feature importance ----

    importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("\nTop 15 features by importance:")
    print(importances.head(15))
    
    return run_id, version, rmse

def promote_if_better(model_name, version, rmse):
    client = mlflow.MlflowClient()
    
    try:
        champion = client.get_model_version_by_alias(model_name, "champion")
        champion_run = client.get_run(champion.run_id)
        champion_rmse = champion_run.data.metrics["rmse"]
    except Exception:
        # no champion set yet — this run becomes the first one
        champion_rmse = float("inf")

    if rmse < champion_rmse:
        client.set_registered_model_alias(model_name, "champion", version)
        print(f"Promoted v{version} (rmse={rmse:.3f}), beat v.prev (rmse={champion_rmse:.3f})")
    else:
        print(f"Kept current champion (rmse={champion_rmse:.3f}); v{version} (rmse={rmse:.3f}) did not improve")

if __name__ == "__main__":
    run_id, version, rmse = train_and_log()
    promote_if_better(MODEL_NAME, version, rmse)
    