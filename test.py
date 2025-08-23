import pandas as pd
import joblib
import numpy as np
import os
import optuna
from sklearn.metrics import r2_score, mean_squared_error
import warnings

warnings.filterwarnings("ignore")

FEATURE_COLUMNS = [
    "IL_10MHz", "IL_50MHz", "IL_100MHz", "IL_500MHz",
    "IL_1000MHz", "IL_5000MHz", "DataRate(Mbps)",
    "preW", "mainW", "postW", "dpre", "dpost", "Rdrv"
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
    meta_model = joblib.load(
        "/Users/nouromran/Documents/upWork/signal integrity optimization/Models/stacked_model.joblib"
    )
    base_models = joblib.load(
        "/Users/nouromran/Documents/upWork/signal integrity optimization/Models/base_models.joblib"
    )

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


# ---------------------------
# Load and prepare the test dataset
# ---------------------------
df_test = pd.read_csv(
    "/Users/nouromran/Documents/upWork/signal integrity optimization/data/Test/35mcab.csv"
)
df_test.drop(columns=["Cable_Name"], inplace=True)

X = df_test.drop(columns=["eyeSNR"])
y = df_test["eyeSNR"].values

# ---------------------------
# Predict for the first sample
# ---------------------------
final_pred, base_preds = load_predict(X.iloc[[0]])
print("Final stacked prediction:", final_pred)
print("Base model predictions:", base_preds)
print("True value:", y[0])

# ---------------------------
# Predict for all samples
# ---------------------------
all_preds = []
for i in range(len(X)):
    pred, _ = load_predict(X.iloc[[i]])
    all_preds.append(pred)

all_preds = np.array(all_preds)

# ---------------------------
# Metrics
# ---------------------------
mae = np.mean(np.abs(all_preds - y))
r2 = r2_score(y, all_preds)
rmse = np.sqrt(mean_squared_error(y, all_preds))

print("Mean Absolute Error:", mae)
print("R2 Score:", r2)
print("Root Mean Squared Error:", rmse)


"""
petersb12.csv
True value: 1.820429
Mean Absolute Error: 0.07730963229854655
R2 Score: 0.9305888463471784
Root Mean Squared Error: 0.11140465583699458



35mcab.csv 

Mean Absolute Error: 0.5315505118470791
R2 Score: 0.1632188430365995
Root Mean Squared Error: 0.7472903458491873
"""