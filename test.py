import pandas as pd
import joblib
import numpy as np
import os 
import optuna

from sklearn.metrics import r2_score, mean_squared_error

import warnings
warnings.filterwarnings("ignore")
FEATURE_COLUMNS = [
    "IL_10MHz", "IL_50MHz", "IL_100MHz", "IL_500MHz", "IL_1000MHz", "IL_5000MHz",
    "DataRate(Mbps)", "preW", "mainW", "postW", "dpre", "dpost", "Rdrv"
]

PARAM_VALUES = {
    "preW":  [0, 2, 5, 7],
    "mainW": [1, 3, 5, 7],
    "postW": [0, 2, 5, 7],
    "dpre":  [0.2, 0.4, 0.6, 0.8],
    "dpost": [0.2, 0.4, 0.6, 0.8],
    "Rdrv":  [300, 700, 1100]
}

def load_predict(X):
    """
    Given X (pd.DataFrame), returns final stacked prediction and base model predictions.
    """
    meta_model = joblib.load("/Users/nouromran/Documents/upWork/signal integrity optimization/Models/stacked_model.joblib")
    base_models = joblib.load("/Users/nouromran/Documents/upWork/signal integrity optimization/Models/base_models.joblib")
    # Predict with base models
    meta_features = []
    base_model_predictions = {}
    for q, model in base_models.items():
        preds = model.predict(X)
        meta_features.append(preds)
        base_model_predictions[f'quantile_{int(q*100)}_pred'] = preds

    # Prepare input for meta model
    quantiles = sorted(base_models.keys())
    meta_feature_names = [f'quantile_pred_{int(q*100)}' for q in quantiles]
    combined_meta_features = pd.DataFrame(
        np.hstack([np.column_stack(meta_features), X]),
        columns=meta_feature_names + FEATURE_COLUMNS
    )

    # Final stacked prediction
    final_prediction = meta_model.predict(combined_meta_features)

    return final_prediction[0], base_model_predictions


df_test = pd.read_csv("/Users/nouromran/Documents/upWork/signal integrity optimization/data/Test/nexa6m.csv")
df_test.drop(columns=["Cable_Name"], inplace=True)

X = df_test.drop(columns=["eyeSNR"]).values
y = df_test["eyeSNR"].values

final_pred, base_preds = load_predict(X[0].reshape(1, -1))
print("Final stacked prediction:", final_pred)
print("Base model predictions:", base_preds)
print("True value:", y[0])

# Optionally, compare for all rows
all_preds = []
for i in range(len(X)):
    pred, _ = load_predict(X[i].reshape(1, -1))
    all_preds.append(pred)
mae = np.mean(np.abs(all_preds - y))
r2 = r2_score(y, all_preds)
rmse = np.sqrt(mean_squared_error(y, all_preds))

print("Mean Absolute Error:", mae)
print("R2 Score:", r2)
print("Root Mean Squared Error:", rmse)