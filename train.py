import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder,MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Ridge , Lasso
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import glob
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
from matplotlib  import  pyplot as plt
from scipy import stats
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
print("Importing libraries completed")
print("*" * 100)
path="/Users/nouromran/Documents/upWork/signal integrity optimization/data/"
if not os.path.exists(path):
    os.makedirs(path)
all_files = glob.glob(os.path.join(path, "*.csv"))
print(f"Found {len(all_files)} CSV files in the directory.")
all_csv = pd.concat(
    (pd.read_csv(f) for f in all_files ),
    ignore_index=True
)
all_csv=all_csv.drop(columns=['Cable_Name','Cable Name'])

X = all_csv.drop(columns=['eyeSNR'])
y = all_csv['eyeSNR']

X_train_base, X_temp, y_train_base, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_train_meta, X_test, y_train_meta, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# --------------------------
# RMSE scorer
# --------------------------
def rmse_scorer_with_nan_handling(y_true, y_pred):
    if np.isnan(y_pred).any() or np.isinf(y_pred).any():
        return float('inf') # Penalize non-finite predictions
    return np.sqrt(mean_squared_error(y_true, y_pred))

rmse_scorer = make_scorer(rmse_scorer_with_nan_handling, greater_is_better=False)

# --------------------------
# Function to grid search for a given quantile
# --------------------------
def gridsearch_lgbm_quantile(X_train, y_train, alpha):
    lgbm = lgb.LGBMRegressor(objective="quantile", alpha=alpha, random_state=42)

    param_grid = {
        "num_leaves":  [31],
        "learning_rate": [0.01],
        "n_estimators":  [1000],
        "min_child_samples":  [20],
        "subsample": [0.8],
        "colsample_bytree": [0.8]
    }

    grid = GridSearchCV(lgbm, param_grid, scoring=rmse_scorer, cv=3, verbose=1, n_jobs=-1, error_score='raise')
    grid.fit(X_train, y_train)

    return grid.best_estimator_

# --------------------------
# Train base quantile models (with tuned params)
# --------------------------
quantiles = [0.5, 0.9, 0.99]
base_models = {}
train_meta_features_list = []
test_meta_features_list = []

for q in quantiles:
    print(f"Training model for quantile: {q}") # Added print statement
    best_model = gridsearch_lgbm_quantile(X_train_base, y_train_base, q)
    base_models[q] = best_model

    # Meta features
    train_meta_features_list.append(best_model.predict(X_train_meta))
    test_meta_features_list.append(best_model.predict(X_test))

# Stack meta features with original inputs and convert to DataFrames
train_meta_features = np.column_stack(train_meta_features_list)
test_meta_features = np.column_stack(test_meta_features_list)

# Create feature names for meta features
meta_feature_names = [f'quantile_pred_{int(q*100)}' for q in quantiles]
original_feature_names = X_train_meta.columns.tolist()
all_feature_names = meta_feature_names + original_feature_names

train_meta_features = np.hstack([train_meta_features, X_train_meta])
test_meta_features = np.hstack([test_meta_features, X_test])

train_meta_features = pd.DataFrame(train_meta_features, columns=all_feature_names, index=X_train_meta.index)
test_meta_features = pd.DataFrame(test_meta_features, columns=all_feature_names, index=X_test.index)


print("Train meta features shape:", train_meta_features.shape) # Added print statement
print("Test meta features shape:", test_meta_features.shape)   # Added print statement


# --------------------------
# Train meta model (simple regression LightGBM)
# --------------------------
meta_model = lgb.LGBMRegressor(objective="regression", learning_rate=0.05, n_estimators=500, random_state=42)
meta_model.fit(train_meta_features, y_train_meta)

# --------------------------
# Evaluate
# --------------------------
stacked_preds = meta_model.predict(test_meta_features)
print("Stacked Model MSE:", mean_squared_error(y_test, stacked_preds))
print("Stacked Model R²:", r2_score(y_test, stacked_preds))

for q, model in base_models.items():
    base_preds = model.predict(X_test)
    print(f"Base Q{int(q*100)} MSE:", mean_squared_error(y_test, base_preds))


# Generate predictions from the stacked model on the test set
stacked_preds_test = meta_model.predict(test_meta_features)

# Generate predictions from each base model on the test set
base_preds_test = {}
for q, model in base_models.items():
    base_preds_test[f'Q{int(q*100)}'] = model.predict(X_test)

# Define the filename for saving the model
model_filename = 'stacked_model.joblib'

# Save the stacked model to the file
joblib.dump(meta_model, model_filename)

print(f"Stacked model saved to {model_filename}")




# Define the filename for saving the base models
base_models_filename = 'base_models.joblib'

# Save the base_models dictionary to the file
joblib.dump(base_models, base_models_filename)

print(f"Base models saved to {base_models_filename}")