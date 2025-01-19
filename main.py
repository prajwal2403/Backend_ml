from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Request, Depends
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from starlette.status import HTTP_422_UNPROCESSABLE_ENTITY

# Data Processing and I/O
from io import BytesIO, StringIO
import pandas as pd
import numpy as np
from scipy import stats
import json
import csv

# Machine Learning
from lazypredict.Supervised import LazyClassifier, LazyRegressor
from sklearn.model_selection import (
    train_test_split, 
    cross_val_score, 
    KFold, 
    StratifiedKFold,
    GridSearchCV,
    RandomizedSearchCV
)
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import (
    LabelEncoder, 
    MinMaxScaler, 
    StandardScaler, 
    RobustScaler, 
    PowerTransformer,
    OneHotEncoder,
    OrdinalEncoder
)
from sklearn.ensemble import (
    VotingClassifier, 
    VotingRegressor, 
    RandomForestClassifier, 
    RandomForestRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor
)
from sklearn.metrics import (
    make_scorer, 
    accuracy_score, 
    r2_score, 
    mean_squared_error, 
    f1_score,
    precision_score, 
    recall_score, 
    roc_auc_score, 
    mean_absolute_error,
    explained_variance_score,
    confusion_matrix,
    classification_report
)
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import (
    SelectKBest,
    f_classif,
    mutual_info_classif,
    RFE
)

# Parallel Processing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import multiprocessing
from joblib import Parallel, delayed

# Data Validation and Models
from pydantic import BaseModel, Field, validator
from typing import Dict, Any, Tuple, List, Optional, Union

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

# System and Utilities
import warnings
import os
import io
import time
import logging
import datetime
from pathlib import Path
import tempfile
import shutil
from functools import lru_cache

# Memory Management
import gc
import psutil

# Large Data Handling
import dask.dataframe as dd

app = FastAPI()

port = int(os.environ.get("PORT", 8000))   # Default to 8000 if PORT is not set


# Allow CORS for frontend
origins = [
    "https://ml-web-zeta.vercel.app"  # Add your frontend URL here
]

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Allow specific origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (e.g., GET, POST, PUT, DELETE)
    allow_headers=["*"],  # Allow all headers
)
class F

class FeatureEngineeringRequest(BaseModel):
    method: str

# Global variable to store uploaded dataset
# Global variables
uploaded_data = None
lazy_predict_results = None
is_data_cleaned = False  # Flag to track if data is cleaned
is_feature_engineering_applied = False  # Flag to track if feature engineering is applied

# Helper function to check dataset existence
def check_uploaded_data():
    if uploaded_data is None:
        raise HTTPException(status_code=400, detail="No dataset uploaded. Please upload a dataset first.")

@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    return JSONResponse(content={"error": str(exc)}, status_code=400)

@app.get("/")
async def testing():
    return "Hello World"

@app.post("/upload/")
async def upload_and_preview_file(file: UploadFile = File(...)):
    global uploaded_data, is_data_cleaned, is_feature_engineering_applied
    if not file:
        return JSONResponse(status_code=400, content={"detail": "No file provided"})
    
    try:
        contents = await file.read()
        file_stream = BytesIO(contents)
        
        if file.filename.endswith(".csv"):
            uploaded_data = pd.read_csv(file_stream)
        elif file.filename.endswith((".xls", ".xlsx")):
            uploaded_data = pd.read_excel(file_stream)
        else:
            return JSONResponse(status_code=400, content={"detail": "Unsupported file type"})
        
        # Reset flags when new data is uploaded
        is_data_cleaned = False
        is_feature_engineering_applied = False
        
        # Convert preview data to JSON serializable format
        preview = uploaded_data.head().replace({np.nan: None}).to_dict(orient="records")
        
        return {
            "status": "success",
            "message": "File uploaded successfully",
            "preview": preview,
            "rows": len(uploaded_data),
            "columns": len(uploaded_data.columns)
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"detail": f"Upload failed: {str(e)}"})

@app.get("/get-data-info/")
def get_data_info():
    check_uploaded_data()
    preview = uploaded_data.head().replace({np.nan: None}).to_dict(orient="records")
    missing_values = uploaded_data.isnull().sum().to_dict()
    
    return {
        "preview": preview,
        "missing_values": missing_values,
        "total_rows": len(uploaded_data),
        "total_columns": len(uploaded_data.columns)
    }

@app.post("/clean-data/")
def clean_data(fill_value: str = None, drop_threshold: float = None):
    global uploaded_data, is_data_cleaned
    check_uploaded_data()
    
    initial_shape = uploaded_data.shape
    if fill_value:
        uploaded_data = uploaded_data.fillna(fill_value)
    if drop_threshold:
        uploaded_data = uploaded_data.dropna(thresh=int(drop_threshold * uploaded_data.shape[1]))
    
    is_data_cleaned = True
    
    return {
        "status": "success",
        "message": "Dataset cleaned successfully",
        "initial_shape": initial_shape,
        "final_shape": uploaded_data.shape,
        "removed_rows": initial_shape[0] - uploaded_data.shape[0],
        "removed_columns": initial_shape[1] - uploaded_data.shape[1]
    }

@app.post("/feature-engineering/")
def feature_engineering(request: FeatureEngineeringRequest):
    global uploaded_data, is_data_cleaned, is_feature_engineering_applied
    check_uploaded_data()
    
    if not is_data_cleaned:
        raise HTTPException(
            status_code=400,
            detail="Data must be cleaned before applying feature engineering. Please call /clean-data/ first."
        )
    
    if request.method not in ["normalize", "standardize"]:
        raise HTTPException(
            status_code=400,
            detail="Invalid method. Supported methods are 'normalize' and 'standardize'."
        )
    
    numeric_cols = uploaded_data.select_dtypes(include="number").columns
    if numeric_cols.empty:
        raise HTTPException(status_code=400, detail="No numeric columns found")
    
    try:
        if request.method == "normalize":
            scaler = MinMaxScaler()
            uploaded_data[numeric_cols] = scaler.fit_transform(uploaded_data[numeric_cols])
            scaling_type = "normalization"
        elif request.method == "standardize":
            scaler = StandardScaler()
            uploaded_data[numeric_cols] = scaler.fit_transform(uploaded_data[numeric_cols])
            scaling_type = "standardization"
        
        is_feature_engineering_applied = True
        
        return {
            "status": "success",
            "message": f"Applied {scaling_type} to {len(numeric_cols)} numeric columns",
            "scaled_columns": numeric_cols.tolist(),
            "preview": uploaded_data[numeric_cols].head().replace({np.nan: None}).to_dict(orient="records")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scaling failed: {str(e)}")

@app.post("/recommend-learning/")
def recommend_learning() -> Dict[str, Any]:
    """
    Recommends the most suitable machine learning approach based on dataset characteristics.
    
    Returns:
        Dict containing recommended learning type and detailed dataset analysis
    """
    def check_time_series_indicators(data: pd.DataFrame) -> bool:
        # Check for datetime columns
        date_columns = data.select_dtypes(include=['datetime64']).columns
        if len(date_columns) > 0:
            # Check if there's a regular time interval
            for date_col in date_columns:
                intervals = pd.Series(data[date_col].sort_values().diff().dropna())
                if len(intervals.value_counts()) <= 3:  # Allow some irregularity
                    return True
        return False
    
    def check_text_data(data: pd.DataFrame) -> bool:
        # Check for text data characteristics
        for col in data.select_dtypes(include=['object']):
            sample = data[col].dropna().astype(str)
            if sample.empty:
                continue
            # Check average word count and string length
            avg_words = sample.str.split().str.len().mean()
            avg_length = sample.str.len().mean()
            if avg_words > 3 or avg_length > 20:
                return True
        return False

    check_uploaded_data()
    data = uploaded_data.copy()
    
    # Enhanced data type analysis
    num_columns = data.select_dtypes(include=['number']).columns
    cat_columns = data.select_dtypes(include=['object', 'category']).columns
    date_columns = data.select_dtypes(include=['datetime64']).columns
    
    total_rows = data.shape[0]
    total_columns = data.shape[1]
    
    # Get target column
    target_column = data.columns[-1]
    target_values = data[target_column]
    
    # Calculate class imbalance for classification
    class_balance = None
    if pd.api.types.is_object_dtype(target_values) or pd.api.types.is_categorical_dtype(target_values):
        value_counts = target_values.value_counts()
        class_balance = (value_counts.min() / value_counts.max()) if not value_counts.empty else None
    
    # Calculate missing values percentage
    missing_percentage = (data.isnull().sum().sum() / (total_rows * total_columns)) * 100
    
    # Enhanced learning type determination
    learning_type = None
    confidence = "high"
    additional_notes = []

    # Check for text data first
    if check_text_data(data):
        learning_type = "Natural Language Processing"
        additional_notes.append("Text data detected - consider text preprocessing and NLP techniques")
    
    # Check for time series
    elif check_time_series_indicators(data):
        learning_type = "Time Series Analysis"
        additional_notes.append("Regular time intervals detected - consider time series specific algorithms")
    
    # Classification check with enhanced criteria
    elif (target_values.nunique() <= 100 and  # Increased threshold for multi-class
          (pd.api.types.is_object_dtype(target_values) or 
           pd.api.types.is_categorical_dtype(target_values) or 
           (pd.api.types.is_numeric_dtype(target_values) and target_values.nunique() <= 100))):
        learning_type = "Classification (Supervised)"
        
        if class_balance and class_balance < 0.2:
            additional_notes.append("Warning: Significant class imbalance detected")
            confidence = "medium"
            
        if target_values.nunique() > 10:
            additional_notes.append("Multi-class classification scenario")
            
        if target_values.nunique() == 2:
            additional_notes.append("Binary classification scenario")
    
    # Regression check with enhanced criteria
    elif (pd.api.types.is_numeric_dtype(target_values) and 
          target_values.nunique() > 100):
        learning_type = "Regression (Supervised)"
        
        # Check for zero variance
        if target_values.std() == 0:
            confidence = "low"
            additional_notes.append("Warning: Target variable has zero variance")
            
        # Check for normal distribution
        if abs(target_values.skew()) > 1:
            additional_notes.append("Note: Target variable shows significant skewness")
    
    # Clustering check with enhanced criteria
    elif (len(num_columns) > 1 and 
          len(cat_columns) / total_columns < 0.3):  # Less than 30% categorical features
        learning_type = "Clustering (Unsupervised)"
        additional_notes.append("Dataset suitable for clustering analysis")
        
        if missing_percentage > 10:
            confidence = "medium"
            additional_notes.append("High percentage of missing values may affect clustering quality")
    
    # Dimensionality reduction check
    elif len(num_columns) > 10:  # High dimensional data
        learning_type = "Dimensionality Reduction"
        additional_notes.append("High dimensional data detected - consider PCA or t-SNE")
    
    else:
        learning_type = "Dataset analysis ambiguous"
        confidence = "low"
        additional_notes.append("Manual inspection recommended")

    return {
        "recommended_learning": learning_type,
        "confidence": confidence,
        "analysis": {
            "total_rows": total_rows,
            "total_columns": total_columns,
            "num_numeric_columns": len(num_columns),
            "num_categorical_columns": len(cat_columns),
            "num_date_columns": len(date_columns),
            "target_unique_values": target_values.nunique(),
            "missing_values_percentage": missing_percentage,
            "class_balance_ratio": class_balance if class_balance is not None else "N/A"
        },
        "additional_notes": additional_notes
    }

class Dataset(BaseModel):
    dataset: list

def process_categorical_column(column_name, series):
    le = LabelEncoder()
    series_cleaned = series.fillna('missing').astype(str)
    transformed = le.fit_transform(series_cleaned)
    return column_name, transformed, le

def convert_to_numeric(df):
    label_encoders = {}
    cat_columns = df.select_dtypes(include=['object', 'category']).columns
    num_columns = df.select_dtypes(include=['int64', 'float64']).columns

    for col in cat_columns:
        col_name, transformed_data, le = process_categorical_column(col, df[col])
        df[col_name] = transformed_data
        label_encoders[col_name] = le

    df[num_columns] = df[num_columns].fillna(df[num_columns].median())
    return df, label_encoders

def get_feature_importance(X, y, is_classification):
    model_cls = RandomForestClassifier if is_classification else RandomForestRegressor
    model = model_cls(n_estimators=50, n_jobs=-1, random_state=42)
    model.fit(X, y)
    return dict(zip(X.columns, model.feature_importances_))

def lazy_predict(data):
    global lazy_predict_results
    if data.empty or len(data.columns) < 2:
        raise ValueError("Invalid dataset")

    data_numeric, label_encoders = convert_to_numeric(data)
    target_column = data_numeric.columns[-1]
    X = data_numeric.drop(columns=[target_column])
    y = data_numeric[target_column]

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    is_classification = len(np.unique(y)) < 10

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42,
        stratify=y if is_classification else None
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        predictor_cls = LazyClassifier if is_classification else LazyRegressor
        predictor = predictor_cls(verbose=0, ignore_warnings=True)
        models, _ = predictor.fit(X_train, X_test, y_train, y_test)

    feature_importance = get_feature_importance(X_scaled, y, is_classification)

    lazy_predict_results = {
        "problem_type": "Classification" if is_classification else "Regression",
        "data_summary": {
            "total_samples": len(X),
            "features": len(X.columns),
            "target_unique_values": len(np.unique(y)),
        },
        "model_results": {
            "best_model": {
                "name": models.index[0],
                "score": models.iloc[0, 0]
            },
            "top_models": models.head(5).reset_index().to_dict(orient="records")
        },
        "feature_importance": {
            k: float(v) for k, v in feature_importance.items()
        }
    }
    return lazy_predict_results

# FastAPI Endpoints
@app.post("/lazy-predict/")
async def lazy_predict_endpoint(background_tasks: BackgroundTasks):
    global lazy_predict_results
    if uploaded_data is None:
        raise HTTPException(status_code=400, detail="Dataset not uploaded yet.")
    result = lazy_predict(uploaded_data)
    background_tasks.add_task(gc.collect)
    return result

@app.get("/visualize/{type}/")
def visualize(type: str):
    global uploaded_data, lazy_predict_results
    check_uploaded_data()
    
    img_buffer = io.BytesIO()
    plt.figure(figsize=(10, 6))
    
    try:
        if type == "data-quality":
            sns.heatmap(uploaded_data.isnull(), cbar=False, cmap="viridis")
            plt.title("Data Quality Analysis (Missing Values Heatmap)")
            
        elif type == "correlation":
            numeric_data = uploaded_data.select_dtypes(include=[np.number])
            if len(numeric_data.columns) > 1:
                sns.heatmap(numeric_data.corr(), annot=True, cmap="coolwarm", fmt=".2f")
                plt.title("Correlation Heatmap")
            else:
                raise HTTPException(status_code=400, detail="Not enough numeric columns for correlation analysis")
                
        elif type == "model-performance" and lazy_predict_results:
            models = lazy_predict_results["model_results"]["top_models"]
            model_names = [model["Model"] for model in models]
            scores = [model["Accuracy"] for model in models]
            sns.barplot(x=model_names, y=scores, palette="Blues_d")
            plt.title("Top Model Performance Metrics")
            plt.xlabel("Model")
            plt.ylabel("Accuracy")
            
        else:
            raise HTTPException(status_code=400, detail="Invalid visualization type or no prediction results available")
            
        plt.tight_layout()
        plt.savefig(img_buffer, format="png")
        plt.close()
        img_buffer.seek(0)
        return StreamingResponse(img_buffer, media_type="image/png")
        
    except Exception as e:
        plt.close()
        raise HTTPException(status_code=500, detail=f"Error creating visualization: {str(e)}")

@app.get("/visualize/feature-importance/")
def visualize_feature_importance():
    global lazy_predict_results
    check_uploaded_data()
    
    if lazy_predict_results is None:
        raise HTTPException(status_code=400, detail="Run /lazy-predict/ first to generate feature importance.")
    
    try:
        feature_importance = lazy_predict_results.get("feature_importance", {})
        if not feature_importance:
            raise HTTPException(status_code=400, detail="No feature importance data available. Run /lazy-predict/ first.")
        
        features = list(feature_importance.keys())
        importance_values = list(feature_importance.values())
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x=importance_values, y=features, palette="viridis")
        plt.title("Feature Importance")
        plt.xlabel("Importance Score")
        plt.ylabel("Features")
        
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format="png")
        plt.close()
        img_buffer.seek(0)
        
        return StreamingResponse(img_buffer, media_type="image/png")
    except Exception as e:
        print(f"Error in /visualize/feature-importance/: {str(e)}")  # Log the error
        raise HTTPException(status_code=500, detail=f"Error creating feature importance visualization: {str(e)}")

@app.get("/visualize/distribution/")
def visualize_distribution(column: str):
    global uploaded_data
    check_uploaded_data()
    
    if column not in uploaded_data.columns:
        raise HTTPException(status_code=400, detail=f"Column '{column}' not found in dataset.")
    
    try:
        plt.figure(figsize=(10, 6))
        if pd.api.types.is_numeric_dtype(uploaded_data[column]):
            sns.histplot(uploaded_data[column], kde=True)
            plt.title(f"Distribution of {column}")
        else:
            sns.countplot(y=uploaded_data[column])
            plt.title(f"Value Counts for {column}")
        
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format="png")
        plt.close()
        img_buffer.seek(0)
        
        return StreamingResponse(img_buffer, media_type="image/png")
    except Exception as e:
        print(f"Error in /visualize/distribution/: {str(e)}")  # Log the error
        raise HTTPException(status_code=500, detail=f"Error creating distribution visualization: {str(e)}")

@app.get("/visualize/word-cloud/")
def visualize_word_cloud(column: str):
    global uploaded_data
    check_uploaded_data()
    
    if column not in uploaded_data.columns:
        raise HTTPException(status_code=400, detail=f"Column '{column}' not found in dataset.")
    
    if not pd.api.types.is_string_dtype(uploaded_data[column]):
        raise HTTPException(status_code=400, detail=f"Column '{column}' must contain text data.")
    
    try:
        text_data = " ".join(uploaded_data[column].dropna().astype(str))
        wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text_data)
        
        plt.figure(figsize=(10, 6))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.title(f"Word Cloud for {column}")
        
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format="png")
        plt.close()
        img_buffer.seek(0)
        
        return StreamingResponse(img_buffer, media_type="image/png")
    except Exception as e:
        print(f"Error in /visualize/word-cloud/: {str(e)}")  # Log the error
        raise HTTPException(status_code=500, detail=f"Error creating word cloud: {str(e)}")

@app.get("/download/")
async def download_cleaned_data():
    global uploaded_data
    check_uploaded_data()
    
    if not is_data_cleaned:
        raise HTTPException(status_code=400, detail="Data must be cleaned before downloading")
    
    try:
        # Convert DataFrame to CSV
        stream = StringIO()
        uploaded_data.to_csv(stream, index=False)
        response = StreamingResponse(
            iter([stream.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=processed_data.csv"}
        )
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating download file: {str(e)}")

# Run the FastAPI app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=port)