import joblib
import pandas as pd
import numpy as np
import optuna


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

def make_feature_vector(il_values, data_rate, preW, mainW, postW, dpre, dpost, rdrv):
    feature_values = il_values + [data_rate, preW, mainW, postW, dpre, dpost, rdrv]
    return pd.DataFrame([feature_values], columns=FEATURE_COLUMNS)

def csv_processing(path,il_values, data_rate):
    """
    Reads a CSV file, processes it, and returns a DataFrame with the specified features.
    """
    df = pd.read_csv(path)
     # Ensure only the relevant columns are kept
    
    df['IL_10MHz'] = il_values[0]
    df['IL_50MHz'] = il_values[1]
    df['IL_100MHz'] = il_values[2]
    df['IL_500MHz'] = il_values[3]
    df['IL_1000MHz'] = il_values[4]
    df['IL_5000MHz'] = il_values[5]
    df['DataRate(Mbps)'] = data_rate
    df["eyeSNR"] = df["value"]
    df["Rdrv"] = df["params_Rdrv"]
    df["dpost"] = df["params_dpost"]
    df["dpre"] = df["params_dpre"]
    df["postW"] = df["params_postW"]
    df["mainW"] = df["params_mainW"]
    df["preW"] = df["params_preW"]
    df.drop(columns=[
        "params_Rdrv", "params_dpost", "params_dpre",
        "params_postW", "params_mainW", "params_preW","value"
    ], inplace=True)
    feature = FEATURE_COLUMNS + ["eyeSNR"]
    print(df["eyeSNR"])
    df = df[feature]
    df.to_csv(path, index=False)  # Save processed data
     # Ensure only the relevant columns are kept
    # Remove the original value column
      # Placeholder for EyeSNR
    return df

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


def optimize_preemphasis(il_values, data_rate, trials=50 ):
    """
    Finds the best discrete pre-emphasis parameters that maximize eyeSNR.
    """
    def objective(trial):
        params = {
            name: trial.suggest_categorical(name, choices)
            for name, choices in PARAM_VALUES.items()
        }
        X_test = make_feature_vector(
            il_values, data_rate,
            params["preW"], params["mainW"], params["postW"],
            params["dpre"], params["dpost"], params["Rdrv"]
        )
        eyesnr, _ = load_predict(X_test)
        return eyesnr  
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=trials)
    df = study.trials_dataframe()
    dataframe_path = "optimization preemphasis/optuna_results.csv"
    df.to_csv(dataframe_path, index=False)
    csv_processing(dataframe_path, il_values, data_rate)
    print("Saved results to optuna_results.csv")
    return study.best_params, study.best_value


if __name__ == "__main__":
    #il_values_example = [-2.45, -2.54, -2.63, -3.14, -3.90, -18.80]  # IL features
    #data_rate_example = 100  # Mbps
    #il_values_example = [-4.89, -7.13, -9.7, -19.95, -22.99, -34.84]
    #data_rate_example = 10 

    il_values_example = [-3.57, -3.7, -3.83, -4.8, -5.93, -30.57]
    data_rate_example = 10  # Mbps
    best_params, best_eyesnr = optimize_preemphasis(il_values_example, data_rate_example, trials=700)
    #print("Best Parameters:", best_params)
    #print("Best EyeSNR:", best_eyesnr)

    # Predict with best parameters
    X_best = make_feature_vector(
        il_values_example, data_rate_example,
        best_params["preW"], best_params["mainW"], best_params["postW"],
        best_params["dpre"], best_params["dpost"], best_params["Rdrv"]
    )
    final_pred, base_preds = load_predict(X_best)
    print("Final Prediction:", final_pred)
    print("Base Model Predictions:", base_preds)

