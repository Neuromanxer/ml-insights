import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, log_loss
import lightgbm as lgb
import xgboost as xgb
from lightgbm import early_stopping, log_evaluation
import catboost as cb
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# XGBoost Parameters for Classification
xgb_params_c = {
    "objective": "binary:logistic",  # For binary classification
    "n_estimators": 2000,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "min_child_weight": 10,
    "random_state": 42,
    "use_label_encoder": False
}

# LightGBM Parameters for Classification
lgb_params_c = {
    "objective": "binary",  # For binary classification
    "n_estimators": 2000,
    "learning_rate": 0.05,
    "max_depth": -1,  # No limit on tree depth
    "num_leaves": 31,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_samples": 20,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
}

# CatBoost Parameters for Classification
cat_params_c = {
    "iterations": 2000,
    "learning_rate": 0.05,
    "depth": 6,
    "l2_leaf_reg": 3.0,
    "random_strength": 1.0,
    "bagging_temperature": 1.0,
    "border_count": 254,
    "loss_function": "Logloss",  # For binary classification
    "random_state": 42,
    "verbose": 0,
}

class ModelClassifyingTrainer:
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
        y_true = self.data['Response']
        y_pred = self.data[['ID']].copy()
        y_pred['prediction'] = np.round(preds).astype(int)  # Convert probabilities to class labels

        accuracy = accuracy_score(y_true, y_pred['prediction'])
        precision = precision_score(y_true, y_pred['prediction'], average='weighted')
        recall = recall_score(y_true, y_pred['prediction'], average='weighted')
        f1 = f1_score(y_true, y_pred['prediction'], average='weighted')

        print(f'Validation Metrics for {title}:')
        print(f'Accuracy: {accuracy:.4f}')
        print(f'Precision: {precision:.4f}')
        print(f'Recall: {recall:.4f}')
        print(f'F1 Score: {f1:.4f}')

    def train_model(self, params, target, title):
        """Train LGBM, XGB, and CatBoost with cross-validation."""
        cv, oof_preds = self._prepare_cv()
        X = self.data.drop(['ID', 'Response'], axis=1)
        y = self.data[target]
        models, fold_scores = [], []

        for fold, (train_index, val_index) in enumerate(cv.split(X, y)):
            print(f"Training Fold {fold + 1}/{self._n_splits}...")

            X_train, X_valid = X.iloc[train_index], X.iloc[val_index]
            y_train, y_valid = y.iloc[train_index], y.iloc[val_index]

            if title.startswith('LightGBM'):
                model = LGBMClassifier(**params)
                model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], eval_metric='logloss', callbacks=[log_evaluation(500)])

            elif title.startswith('CatBoost'):
                model = CatBoostClassifier(**params)
                model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=500)

            elif title.startswith('XGBoost'):
                model = XGBClassifier(**params)
                model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=500)

            models.append(model)
            y_pred = model.predict(X_valid)  # Class predictions
            oof_preds[val_index] = y_pred

            # Compute all classification metrics
            fold_accuracy = accuracy_score(y_valid, y_pred)
            fold_precision = precision_score(y_valid, y_pred, average='weighted')
            fold_recall = recall_score(y_valid, y_pred, average='weighted')
            fold_f1 = f1_score(y_valid, y_pred, average='weighted')

            print(f"Fold {fold + 1} Metrics:")
            print(f"  Accuracy: {fold_accuracy:.4f}")
            print(f"  Precision: {fold_precision:.4f}")
            print(f"  Recall: {fold_recall:.4f}")
            print(f"  F1 Score: {fold_f1:.4f}")

            fold_scores.append((fold_accuracy, fold_precision, fold_recall, fold_f1))

        print(f"Training Complete for {title}!")
        return models, oof_preds