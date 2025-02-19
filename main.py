import logging
import time
import io
import joblib
import pandas as pd
from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import FileResponse, JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from regression import ModelTrainer, lgb_params, cat_params, xgb_params
from clustering import run_kmeans
from classification import ModelClassifyingTrainer, lgb_params_c, cat_params_c, xgb_params_c
from preprocessing import preprocess_data
from feature_importance import plot_feature_importance, plot_shap_explanations, plot_partial_dependence, plot_pdp, plot_counterfactual
from sklearn.metrics import mean_squared_error, accuracy_score
from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel
from passlib.context import CryptContext
from sqlalchemy.orm import Session
from database import SessionLocal, engine, User
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s",
    handlers=[
        logging.FileHandler("app.log"),  # Save logs to a file
        logging.StreamHandler()  # Print logs to console
    ],
)
logger = logging.getLogger(__name__)

app = FastAPI()

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

class UserCreate(BaseModel):
    email: str
    password: str

@app.post("/register/")
def register(user: UserCreate, db: Session = Depends(get_db)):
    hashed_password = pwd_context.hash(user.password)
    db_user = User(email=user.email, password=hashed_password)
    db.add(db_user)
    db.commit()
    return {"message": "User registered successfully"}

# Middleware for logging requests and responses
class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        body = await request.body()

        logger.info(f"📥 Request: {request.method} {request.url} - Body: {body.decode('utf-8')}")
        
        response = await call_next(request)
        process_time = time.time() - start_time

        logger.info(f"📤 Response: {response.status_code} - Time: {process_time:.4f}s")
        
        return response

app.add_middleware(LoggingMiddleware)

# Exception handler to log errors
@app.exception_handler(Exception)
async def exception_handler(request: Request, exc: Exception):
    logger.error(f"❌ Exception: {request.method} {request.url} - Error: {str(exc)}")
    return JSONResponse(status_code=500, content={"message": "Internal Server Error"})

@app.get("/")
def read_root():
    logger.info("✅ Root endpoint accessed")
    return {"message": "FastAPI is running"}

@app.post("/cluster/")
async def cluster(file: UploadFile = File(...), drop_columns: str = Form("")):
    try:
        df = pd.read_csv(io.BytesIO(await file.read()))
        drop_cols = drop_columns.split(",") if drop_columns else []

        clustered_df, model, k = run_kmeans(df, drop_cols)
        joblib.dump(model, "kmeans_model.pkl")

        logger.info(f"✅ KMeans clustering completed with K={k}")
        return {"message": f"KMeans completed with K={k}", "clusters": clustered_df.to_dict()}
    except Exception as e:
        logger.error(f"❌ Error in /cluster: {str(e)}")
        return JSONResponse(status_code=500, content={"message": "Error in clustering"})

@app.post("/regression/")
async def regression(file: UploadFile = File(...), target_column: str = Form("target"), drop_columns: str = Form("")):
    try:
        df = pd.read_csv(io.BytesIO(await file.read()))
        preprocess_data(df)
        drop_cols = drop_columns.split(",") if drop_columns else []

        trainer = ModelTrainer(df)

        lgb_model, lgb_oof_preds = trainer.train_model(lgb_params, target_column, title="LightGBM")
        ctb_model, ctb_oof_preds = trainer.train_model(cat_params, target_column, title='CatBoost')
        xgb_model , xgb_oof_preds = trainer.train_model(xgb_params, target_column, title='XGBoost')

        y_true = df[target_column]

        results = {
            "LightGBM": mean_squared_error(y_true, lgb_oof_preds, squared=False),
            "CatBoost": mean_squared_error(y_true, ctb_oof_preds, squared=False),
            "XGBoost": mean_squared_error(y_true, xgb_oof_preds, squared=False),
        }

        best_model_name = min(results, key=results.get)
        best_model = {"LightGBM": lgb_model, "CatBoost": ctb_model, "XGBoost": xgb_model}[best_model_name]
        joblib.dump(best_model, "best_regressor.pkl")

        plot_feature_importance(best_model[0], df.drop(columns=[target_column,'ID']))  
        plot_shap_explanations(best_model[0], df.drop(columns=[target_column, 'ID']))

                # Calculate target range and spread
        target_min = df[target_column].min()
        target_max = df[target_column].max()
        target_std = df[target_column].std()

        # Construct RMSE evaluation message
        rmse_analysis = (
            f"📊 Target Range: {target_min} to {target_max} | Std Dev: {target_std:.2f}\n"
            f"📉 RMSE Scores: {results}\n"
            f"🔍 Interpretation: If RMSE is much lower than Std Dev, the model is making useful predictions. "
            f"If RMSE is close to Std Dev, the model might not be adding much value over a simple mean prediction."
        )

        logger.info(f"✅ Regression completed | Best model: {best_model} | {rmse_analysis}")
        return {"status": "success", "best_model": best_model, "rmse_analysis": rmse_analysis}

    except Exception as e:
        logger.error(f"❌ Error in /regression: {str(e)}")
        return JSONResponse(status_code=500, content={"message": "Error in regression"})

@app.post("/classification/")
async def classification(file: UploadFile = File(...), target_column: str = Form("target"), drop_columns: str = Form("")):
    try:
        df = pd.read_csv(io.BytesIO(await file.read()))
        preprocess_data(df)
        drop_cols = drop_columns.split(",") if drop_columns else []

        trainer = ModelClassifyingTrainer(df)

        lgb_model, lgb_oof_preds = trainer.train_model(lgb_params_c, target_column, title="LightGBM")
        ctb_model, ctb_oof_preds = trainer.train_model(cat_params_c, target_column, title='CatBoost')
        xgb_model, xgb_oof_preds = trainer.train_model(xgb_params_c, target_column, title='XGBoost')

        y_true = df[target_column]

        results = {
            "LightGBM": accuracy_score(y_true, lgb_oof_preds),
            "CatBoost": accuracy_score(y_true, ctb_oof_preds),
            "XGBoost": accuracy_score(y_true, xgb_oof_preds),
        }

        best_model_name = max(results, key=results.get)
        best_model = {"LightGBM": lgb_model, "CatBoost": ctb_model, "XGBoost": xgb_model}[best_model_name]
        # Generate and save feature importance plot
        plot_shap_explanations(best_model[0], df.drop(columns=[target_column, 'ID']))
        plot_feature_importance(best_model[0], df.drop(columns=[target_column, 'ID']))

        joblib.dump(best_model, "best_classifier.pkl")



        logger.info(f"✅ Classification completed | Best model: {best_model_name} | Accuracy Scores: {results}")
        plot_feature_importance(best_model[0], df.drop(columns=[col for col in ['ID', target_column] if col in df]))
        return {"message": "Classification completed", "best_model": best_model_name, "accuracy_scores": results}
    except Exception as e:
        logger.error(f"❌ Error in /classification: {str(e)}")
        return JSONResponse(status_code=500, content={"message": "Error in classification"})

import io
import time
import joblib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from fastapi import FastAPI, UploadFile, File, Form
from sklearn.inspection import partial_dependence


@app.post("/visualize/")
async def visualize(
    file: UploadFile = File(...), 
    target_column: str = Form(...), 
    feature_column: str = Form(...)
):
    # Read the uploaded CSV file
    df = pd.read_csv(io.BytesIO(await file.read()))
    df, CATS, NUMS = preprocess_data(df, RMV=["ID", target_column])
    # Check if the specified columns exist
    if target_column not in df.columns or feature_column not in df.columns:
        return {"error": "One or more specified columns not found in the dataset."}

    # Define feature set
    RMV = ["ID", target_column]
    FEATURES = [c for c in df.columns if c not in RMV]  

    if feature_column not in FEATURES:
        return {"error": f"Feature '{feature_column}' is not valid for modeling."}

    feature_index = df.columns.get_loc(feature_column)  # Get correct feature index

    # Load the trained model
    best_model = joblib.load("best_classifier.pkl")
    single_model = best_model[0]  # Pick the first model
    
    # Generate unique file names to avoid overwriting
    timestamp = int(time.time())  # Unique identifier
    scatter_path = f"scatter_plot_{timestamp}.png"
    pdp_path = f"pdp_plot_{timestamp}.png"

    # Generate scatter plot
    plt.figure(figsize=(8, 5))
    if feature_column in CATS:  
        # Box plot for categorical features
        sns.boxplot(x=df[feature_column], y=df[target_column])
        plt.xlabel(feature_column)
        plt.ylabel(target_column)
        plt.title(f"Box Plot: {feature_column} vs {target_column}")
    else:  
        # Scatter plot for numerical features
        sns.scatterplot(x=df[feature_column], y=df[target_column])
        plt.xlabel(feature_column)
        plt.ylabel(target_column)
        plt.title(f"Scatter Plot: {feature_column} vs {target_column}")

    plt.savefig(viz_path, format="png", dpi=300)
    plt.close()

    # Compute Partial Dependence Plot (only for numerical features)
    if feature_column in NUMS:
        feature_index = df.columns.get_loc(feature_column)
        pdp_results = partial_dependence(single_model, df[NUMS], features=[feature_index])
        feature_values = pdp_results.grid_values[0]
        pdp_values = pdp_results.average[0]

        # Plot PDP
        plt.figure(figsize=(8, 5))
        plt.plot(feature_values, pdp_values, color="red", lw=2)
        plt.xlabel(feature_column)
        plt.ylabel("Predicted Target")
        plt.title(f"Partial Dependence Plot for {feature_column}")
        plt.grid()
        plt.savefig(pdp_path, format="png", dpi=300)
        plt.close()
    else:
        pdp_path = None  # Skip PDP for categorical features

    # Compute Partial Dependence Plot
    pdp_results = partial_dependence(single_model, df[FEATURES], features=[feature_index])
    feature_values = pdp_results.grid_values[0]
    pdp_values = pdp_results.average[0]

    # Plot PDP
    plt.figure(figsize=(8, 5))
    plt.plot(feature_values, pdp_values, color="red", lw=2)
    plt.xlabel(feature_column)
    plt.ylabel("Predicted Target")
    plt.title(f"Partial Dependence Plot for {feature_column}")
    plt.grid()
    plt.savefig(pdp_path, format="png", dpi=300)
    plt.close()
    plot_counterfactual(single_model, df[FEATURES], )
    return {
        "message": "Visualization generated successfully",
        "scatter_plot": scatter_path,
        "pdp_plot": pdp_path
    }

from fastapi.responses import JSONResponse
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from my_model_module import train_best_anomaly_detection  


@app.post("/anomaly_detection/")
async def anomaly_detection(file: UploadFile = File(...), target_column: str = Form("target"), drop_columns: str = Form("")):
    try:
        df = pd.read_csv(io.BytesIO(await file.read()))
        drop_cols = drop_columns.split(",") if drop_columns else []

        # Train anomaly detection models
        best_model, best_model_name, results, predictions = train_best_anomaly_detection(df, target_column, drop_cols)

        # Save the best model
        joblib.dump(best_model, "best_anomaly_detection_model.pkl")

        # Compute SHAP values (feature importance)
        explainer = shap.Explainer(best_model)
        shap_values = explainer(df.drop(columns=[target_column] + drop_cols, errors='ignore'))

        # Plot Feature Importance
        shap.summary_plot(shap_values, df.drop(columns=[target_column] + drop_cols, errors='ignore'), show=False)
        plt.savefig("shap_summary.png")

        # PCA for 2D anomaly visualization
        scaler = StandardScaler()
        df_scaled = scaler.fit_transform(df.drop(columns=[target_column] + drop_cols, errors='ignore'))
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(df_scaled)
        df["pca_1"], df["pca_2"] = pca_result[:, 0], pca_result[:, 1]

        # Create a scatter plot of anomalies vs. normal instances
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=df["pca_1"], y=df["pca_2"], hue=predictions, palette={0: "blue", 1: "red"}, alpha=0.6)
        plt.title("Anomaly Detection Visualization")
        plt.legend(title="Class", labels=["Normal", "Anomalous"])
        plt.savefig("anomaly_pca_plot.png")

        # Identify the top 5 most anomalous points
        anomaly_scores = best_model.decision_function(df.drop(columns=[target_column] + drop_cols, errors='ignore'))
        df["Anomaly_Score"] = anomaly_scores
        top_anomalies = df.nlargest(5, "Anomaly_Score")

        return {
            "message": "Anomaly detection completed",
            "best_model": best_model_name,
            "evaluation_scores": results,
            "top_anomalies": top_anomalies.to_dict(orient="records"),
            "feature_importance_plot": "shap_summary.png",
            "anomaly_visualization": "anomaly_pca_plot.png",
        }
    
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": "Error in anomaly detection", "error": str(e)})
from fastapi import FastAPI, UploadFile, File, Form
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
from authlib.integrations.starlette_client import OAuth
app = FastAPI()

oauth = OAuth()

oauth.register(
    name="google",
    client_id="YOUR_CLIENT_ID",
    client_secret="YOUR_CLIENT_SECRET",
    authorize_url="https://accounts.google.com/o/oauth2/auth",
    authorize_params={"scope": "email profile"},
    access_token_url="https://oauth2.googleapis.com/token",
    client_kwargs={"scope": "email profile"},
)

@app.get("/login/google")
async def login_google(request: Request):
    redirect_uri = "http://localhost:8000/auth"
    return await oauth.google.authorize_redirect(request, redirect_uri)

# 📊 Feature Importance Analysis
@app.post("/feature_impact/")
async def feature_impact(file: UploadFile = File(...), target_column: str = Form(...)):
    df = pd.read_csv(file.file)
    model = joblib.load("best_classifier.pkl")  # Load best trained model
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(df.drop(columns=[target_column]))

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, df.drop(columns=[target_column]))
    plt.savefig("feature_importance.png")
    return {"message": "Feature Importance Generated"}

# 🔄 Counterfactual Analysis
@app.post("/counterfactual/")
async def counterfactual(file: UploadFile = File(...), target_column: str = Form(...)):
    df = pd.read_csv(file.file)
    model = joblib.load("best_classifier.pkl")
    sample = df.sample(1)  # Pick a random sample

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(df.drop(columns=[target_column]))

    plt.figure(figsize=(10, 6))
    shap.force_plot(explainer.expected_value, shap_values[0], df.drop(columns=[target_column]).iloc[0])
    plt.savefig("counterfactual_analysis.png")
    return {"message": "Counterfactual Analysis Generated"}

# 📌 Customer Segment Analysis
@app.post("/segment_analysis/")
async def segment_analysis(file: UploadFile = File(...), target_column: str = Form(...)):
    from sklearn.cluster import KMeans

    df = pd.read_csv(file.file)
    X = df.drop(columns=[target_column])
    kmeans = KMeans(n_clusters=3, random_state=42).fit(X)
    df["customer_segment"] = kmeans.labels_
    
    return df[["customer_segment"]].to_dict()

# ⚠️ Risk Analysis
@app.post("/risk_analysis/")
async def risk_analysis(file: UploadFile = File(...), target_column: str = Form(...)):
    df = pd.read_csv(file.file)
    model = joblib.load("best_classifier.pkl")
    df["churn_prob"] = model.predict_proba(df.drop(columns=[target_column]))[:, 1]

    plt.figure(figsize=(10, 6))
    df["churn_prob"].hist(bins=20)
    plt.xlabel("Churn Probability")
    plt.ylabel("Count")
    plt.title("Risk Analysis: High Churn Probability Customers")
    plt.savefig("risk_analysis.png")
    
    return {"message": "Risk Analysis Generated"}

# 🛤️ Decision Paths
@app.post("/decision_paths/")
async def decision_paths(file: UploadFile = File(...), target_column: str = Form(...)):
    from sklearn.tree import DecisionTreeClassifier
    from sklearn import tree

    df = pd.read_csv(file.file)
    X, y = df.drop(columns=[target_column]), df[target_column]
    clf = DecisionTreeClassifier(max_depth=4)
    clf.fit(X, y)

    plt.figure(figsize=(20, 10))
    tree.plot_tree(clf, feature_names=X.columns, class_names=["Negative", "Positive"], filled=True)
    plt.savefig("decision_tree.png")

    return {"message": "Decision Paths Generated"}
