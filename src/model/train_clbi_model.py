"""
train_clbi_model.py - CLBI Production Model Training Pipeline

Trains the production Candidate Line Breach Intelligence (CLBI) Logistic Regression model
on the documented training dataset split (3,050 segments across August Complex, Carr Fire, Creek Fire)
and serializes the fitted pipeline artifact and training metadata.

Outputs:
  - models/clbi_logistic_regression.joblib
  - models/clbi_model_metadata.json
"""

import os
import json
from datetime import datetime
import numpy as np
import pandas as pd
import joblib
import sklearn

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

CLEAN_FEATURES = [
    "slope",
    "elevation",
    "distance_to_fire",
    "barrier_width_m",
    "burn_prob",
    "attack_angle",
    "attack_dot_product"
]

def train_production_model():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_path = os.path.join(base_dir, "data", "final", "fireline_segments_v1.parquet")
    models_dir = os.path.join(base_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    print("=" * 80)
    print("CLBI PRODUCTION MODEL TRAINING PIPELINE")
    print("=" * 80)
    print(f"Loading benchmark dataset: {data_path}")
    
    df = pd.read_parquet(data_path)
    train_df = df[df['split'] == 'train'].copy()
    
    n_train = len(train_df)
    train_fires = sorted(train_df['fire_id'].unique().tolist())
    val_fires   = sorted(df[df['split'] == 'val']['fire_id'].unique().tolist())
    test_fires  = sorted(df[df['split'] == 'test']['fire_id'].unique().tolist())
    
    print(f"Training Rows: {n_train} segments")
    print(f"Training Fires ({len(train_fires)}): {train_fires}")
    print(f"Validation Fires ({len(val_fires)}): {val_fires}")
    print(f"Test Fires ({len(test_fires)}): {test_fires}")
    print(f"Model Features ({len(CLEAN_FEATURES)}): {CLEAN_FEATURES}")
    
    X_train = train_df[CLEAN_FEATURES].values
    y_train_held = train_df['label'].values # 1 = Held, 0 = Burned Over
    
    # Standardized 7-Feature Logistic Regression Pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=1000, random_state=42, class_weight=None))
    ])
    
    print("\nFitting Logistic Regression model...")
    pipeline.fit(X_train, y_train_held)
    
    scaler = pipeline.named_steps['scaler']
    clf = pipeline.named_steps['clf']
    
    # Breach Log-Odds Coefficients (-coef for label=0 breach)
    breach_coefs = (-clf.coef_[0]).tolist()
    breach_intercept = float(-clf.intercept_[0])
    coef_dict = dict(zip(CLEAN_FEATURES, [round(c, 4) for c in breach_coefs]))
    
    print("\nFitted Production Model Parameters (Breach Log-Odds):")
    print(f"  Intercept: {breach_intercept:+.4f}")
    for feat, coef in coef_dict.items():
        print(f"  - {feat:<20}: {coef:+.4f}")
        
    # Serialize model artifact
    model_path = os.path.join(models_dir, "clbi_logistic_regression.joblib")
    joblib.dump(pipeline, model_path)
    print(f"\nSaved production model artifact to: {model_path}")
    
    # Serialize model metadata
    metadata = {
        "model_name": "Candidate Line Breach Intelligence (CLBI) Production Model",
        "version": "v1.0.0",
        "model_type": "Standardized 7-Feature Logistic Regression Pipeline (StandardScaler + LogisticRegression)",
        "features": CLEAN_FEATURES,
        "preprocessing": {
            "scaler": "StandardScaler",
            "means": [round(m, 4) for m in scaler.mean_.tolist()],
            "scales": [round(s, 4) for s in scaler.scale_.tolist()]
        },
        "target_definition": {
            "binary_target": "label (1 = Held, 0 = Burned Over)",
            "primary_output": "breach_probability = P(Burned Over) = 1.0 - P(Held)"
        },
        "training_dataset": "data/final/fireline_segments_v1.parquet",
        "training_rows": n_train,
        "training_fires": train_fires,
        "excluded_validation_fires": val_fires,
        "excluded_test_fires": test_fires,
        "breach_coefficients": coef_dict,
        "breach_intercept": round(breach_intercept, 4),
        "sklearn_version": sklearn.__version__,
        "random_state": 42,
        "creation_timestamp": datetime.now().isoformat(),
        "source_experiment": "src/evaluation/baseline_experiment.py",
        "risk_thresholds": {
            "LOW": "< 0.30",
            "MEDIUM": "0.30 to < 0.60",
            "HIGH": "0.60 to < 0.80",
            "CRITICAL": ">= 0.80"
        }
    }
    
    meta_path = os.path.join(models_dir, "clbi_model_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved production model metadata to: {meta_path}")

if __name__ == "__main__":
    train_production_model()
