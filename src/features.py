import numpy as np

def prepare_X(X):
    X = X.copy()
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
        X[c] = X[c].astype("category")
        
    return X, cat_cols