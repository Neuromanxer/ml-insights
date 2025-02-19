import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from lightgbm import log_evaluation
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

# XGBoost Parameters (Basic)
xgb_params = {
    "objective": "reg:squarederror",
    "n_estimators": 2000,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "min_child_weight": 10,
    "random_state": 42,
}

# LightGBM Parameters (Basic)
lgb_params = {
    "objective": "regression",
    "n_estimators": 2000,
    "learning_rate": 0.05,
    "max_depth": -1,  # No limit on tree depth
    "num_leaves": 31,  # Standard number of leaves
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_samples": 20,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
}

# CatBoost Parameters (Basic)
cat_params = {
    "iterations": 2000,
    "learning_rate": 0.05,
    "depth": 6,
    "l2_leaf_reg": 3.0,
    "random_strength": 1.0,
    "bagging_temperature": 1.0,
    "border_count": 254,
    "loss_function": "RMSE",
    "random_state": 42,
    "verbose": 0,
}


class ModelTrainer:
    def __init__(self, data, n_splits=5):
        self.data = data
        self._n_splits = n_splits
        self._length = len(data)

    def _prepare_cv(self):
        """Initialize K-Fold cross-validation."""
        oof_preds = np.zeros(self._length)  # Out-of-fold predictions
        cv = KFold(n_splits=self._n_splits, shuffle=True, random_state=42)
        return cv, oof_preds
    
    def validate_model(self, preds, title):
        """Evaluate model performance using RMSE, MAE, and R²."""
        y_true = self.data['Response']  # Ensure you're using the correct target column

        mse = mean_squared_error(y_true, preds)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, preds)
        r2 = r2_score(y_true, preds)

        print(f"Model Evaluation - {title}:")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE: {mae:.4f}")
        print(f"  R² Score: {r2:.4f}\n")

    def train_model(self, params, target, title):
        """Train LGBM, XGB, and CatBoost with cross-validation."""
        cv, oof_preds = self._prepare_cv()
        oof_metric = self.data[['ID', target]].copy()    
        X = self.data.drop(['ID', target], axis=1)
        y = self.data[target]
        models, fold_scores = [], []
        
        for fold, (train_index, val_index) in enumerate(cv.split(X, y)):
            print(f"Training Fold {fold + 1}/{self._n_splits}...")

            # Split data
            X_train, X_valid = X.iloc[train_index], X.iloc[val_index]
            y_train, y_valid = y.iloc[train_index], y.iloc[val_index]

            if title.startswith('LightGBM'):
                model = lgb.LGBMRegressor(**params)
                model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], eval_metric='rmse', callbacks=[log_evaluation(500)])
            
            elif title.startswith('CatBoost'):
                model = CatBoostRegressor(**params)
                model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=500)
                    
            elif title.startswith('XGBoost'):
                model = XGBRegressor(**params)
                model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=500)
            
            models.append(model) 
            y_pred = model.predict(X_valid)  # No rounding or type conversion

            oof_preds[val_index] = y_pred
            rmse = np.sqrt(mean_squared_error(y_valid, y_pred))
            print(f"RMSE: {rmse:.4f}")
            fold_scores.append(rmse)
        
        print(f"Training Complete for {title}!")
        return models, oof_preds
