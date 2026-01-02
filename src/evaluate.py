import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

def naive_last_value(X, target_col: int = 0):
    return X[:, -1, target_col]

def compute_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)   # no squared arg
    rmse = float(np.sqrt(mse))
    return mae, rmse

