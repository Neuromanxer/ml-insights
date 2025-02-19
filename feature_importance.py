import shap
import matplotlib.pyplot as plt
import pandas as pd
from catboost import CatBoostRegressor, CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
import logging
import dice_ml
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.inspection import partial_dependence, PartialDependenceDisplay
logger = logging.getLogger(__name__)

def plot_feature_importance(model, X):
    """Plots feature importance for tree-based models (RandomForest, XGBoost, LightGBM, CatBoost)"""
    features = X.columns

    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
    elif isinstance(model, (CatBoostRegressor, CatBoostClassifier)):
        importance = model.get_feature_importance()
    else:
        logger.warning("⚠️ Feature importance not available for this model.")
        return
    
    importance_df = pd.DataFrame({"Feature": features, "Importance": importance})
    importance_df = importance_df.sort_values(by="Importance", ascending=False)

    # Log the top 10 most important features
    logger.info(f"📊 Top 10 Features: {importance_df.head(10).to_dict(orient='records')}")

    # Plot feature importance
    plt.figure(figsize=(10, 5))
    plt.barh(importance_df["Feature"], importance_df["Importance"], color="skyblue")
    plt.xlabel("Feature Importance")
    plt.ylabel("Feature")
    plt.title("Feature Importance Plot")
    plt.gca().invert_yaxis()
    plt.savefig("feature_importance.png")
    plt.close()

def plot_shap_explanations(model, X_train):
    """Generates SHAP explanations for tree-based models"""
    
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_train)

    # Log mean absolute SHAP values for top features
    mean_shap_values = pd.DataFrame(
        {"Feature": X_train.columns, "Mean SHAP": abs(shap_values.values).mean(axis=0)}
    ).sort_values(by="Mean SHAP", ascending=False)
    
    logger.info(f"🔍 Mean SHAP Values: {mean_shap_values.head(10).to_dict(orient='records')}")

    # Summary plot (global explanation)
    shap.summary_plot(shap_values, X_train, show=False)
    plt.savefig("shap_summary.png")  # Save instead of show
    plt.close()

    # Waterfall plot (individual explanation)
    shap.waterfall_plot(shap_values[0], show=False)
    plt.savefig("shap_waterfall.png")
    plt.close()

    logger.info("✅ SHAP explanations generated and saved as images.")

def plot_counterfactual(model, X, sample):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    shap.force_plot(explainer.expected_value, shap_values[sample], X.iloc[sample])

import numpy as np
import matplotlib.pyplot as plt
from sklearn.inspection import partial_dependence, PartialDependenceDisplay

def plot_partial_dependence(model, X, features, feature_names, grid_resolution=50):
    
    if isinstance(X, np.ndarray) and feature_names is None:
        raise ValueError("feature_names must be provided when X is a numpy array.")
    
    # Compute Partial Dependence
    pd_results = partial_dependence(model, X, features=features, grid_resolution=grid_resolution)
    
    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    PartialDependenceDisplay(pd_results, features=features, feature_names=feature_names or X.columns).plot(ax=ax)
    plt.show()
    
# Example Usage:
# plot_partial_dependence(trained_model, X_train, features=[0, 1], feature_names=X_train.columns)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.inspection import partial_dependence, PartialDependenceDisplay

def plot_pdp(model, X, feature_name):
    """
    Plots Partial Dependence for a given feature using an already trained model.

    Parameters:
    - model: Trained regressor/classifier (XGBoost, CatBoost, or LightGBM)
    - X: Feature matrix (DataFrame)
    - feature_name: Name of the feature to analyze
    """
    feature_idx = list(X.columns).index(feature_name)  # Get feature index

    pdp_results = partial_dependence(model, X, features=[feature_idx])
    pdp_values = pdp_results.average[0]
    feature_values = pdp_results.grid_values[0]

    plt.figure(figsize=(8, 5))
    plt.plot(feature_values, pdp_values, color="red", lw=2)
    plt.xlabel(feature_name)
    plt.ylabel("Predicted Target")
    plt.title(f"Partial Dependence Plot for {feature_name}")
    plt.grid()
    plt.show()
    
    print(f"📈 PDP shows how '{feature_name}' impacts predictions.")

# Example usage:
# plot_pdp(trained_model, X, "feature_name")
