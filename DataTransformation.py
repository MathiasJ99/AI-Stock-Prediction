#import DataProcessing
import pandas as pd
import numpy as np
from sklearn.feature_selection import RFECV, RFE
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split


def FeatureSelection():
    """Main function to perform feature selection using RFE and RFECV."""
    #df = DataProcessing.GetData()
    df = pd.read_excel("MergedDF.xlsx")

    # Drop the 'Dates' column (if not needed)
    if 'Dates' in df.columns:
        df = df.drop(columns=['Dates'])

    # Check for NaN or infinite values in the dataset
    if df.isnull().any().any() or np.isinf(df.select_dtypes(include=[np.number])).any().any():
        raise ValueError("❌ Dataset contains NaN or infinite values. Please clean the data first.")

    # Define Features (X) and Target (y)
    X = df.drop(columns=["Close", "Adj Close"])
    y = df["Close"]

    # Ensure all features are numeric
    if not X.select_dtypes(include=[np.number]).shape[1] == X.shape[1]:
        raise ValueError("❌ Non-numeric features detected. Please encode categorical features.")

    ''' 
    # Split Data into Train/Test Sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Perform feature selection with RFECV
    try:
        rf_selected = RandomForestFeatureSelection(X_train, y_train)
        xgb_selected = XGBoostFeatureSelection(X_train, y_train)

        print("--------------------------------------------------")
        print(f"📊 Optimal Random Forest Features: {rf_selected}")
        print(f"📊 Optimal XGBoost Features: {xgb_selected}")
    except Exception as e:
        print(f"❌ Error during feature selection: {e}")
    '''

# **Optimal Feature Selection using RFECV**
def optimal_feature_selection(model, X_train, y_train):
    """Uses RFECV to find the optimal number of features."""
    try:
        rfecv = RFECV(estimator=model, step=1, cv=5, scoring='r2')  # Use R² as scoring metric
        rfecv.fit(X_train, y_train)

        selected_features = X_train.columns[rfecv.support_]
        print(f"✅ Optimal number of features for {type(model).__name__}: {rfecv.n_features_}")
        return selected_features
    except Exception as e:
        print(f"❌ Error in optimal_feature_selection: {e}")
        return []


# **Function to Perform RFE**
def perform_rfe(model, X_train, y_train, num_features=8):
    """Performs standard RFE to select a fixed number of features."""
    try:
        rfe = RFE(model, n_features_to_select=num_features)
        rfe.fit(X_train, y_train)

        selected_features = X_train.columns[rfe.support_]
        return selected_features
    except Exception as e:
        print(f"❌ Error in perform_rfe: {e}")
        return []


# **Random Forest Feature Selection**
def RandomForestFeatureSelection(X_train, y_train, use_rfecv=True):
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

    if use_rfecv:
        return optimal_feature_selection(rf_model, X_train, y_train)  # Uses RFECV
    else:
        return perform_rfe(rf_model, X_train, y_train, num_features=8)  # Uses standard RFE


# **XGBoost Feature Selection**
def XGBoostFeatureSelection(X_train, y_train, use_rfecv=True):
    xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)

    if use_rfecv:
        return optimal_feature_selection(xgb_model, X_train, y_train)  # Uses RFECV
    else:
        return perform_rfe(xgb_model, X_train, y_train, num_features=8)  # Uses standard RFE


def DimensionalityReduction():
    """Placeholder for Autoencoder-based feature reduction."""
    pass


def DataNormalisation():
    """Placeholder for Data Normalization."""
    pass


# Run Feature Selection
FeatureSelection()