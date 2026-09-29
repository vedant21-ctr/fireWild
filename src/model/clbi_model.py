"""
clbi_model.py - Candidate Line Breach Intelligence (CLBI) Decision Engine

Production inference engine for fireline breach risk stratification, vulnerability ranking,
explainable feature attribution, and resource prioritization. Loads a pre-trained serialized model.

Features:
  - slope
  - elevation
  - distance_to_fire
  - barrier_width_m
  - burn_prob
  - attack_angle
  - attack_dot_product

Validated Thresholds:
  - LOW:      P(Breach) < 0.30
  - MEDIUM:   0.30 <= P(Breach) < 0.60
  - HIGH:     0.60 <= P(Breach) < 0.80
  - CRITICAL: P(Breach) >= 0.80
"""

import os
import json
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
import joblib

CLEAN_FEATURES = [
    "slope",
    "elevation",
    "distance_to_fire",
    "barrier_width_m",
    "burn_prob",
    "attack_angle",
    "attack_dot_product"
]

FEATURE_LABELS = {
    "attack_dot_product": "Direct Head-Fire Attack Alignment (cos theta)",
    "burn_prob": "Predicted Next-Day Spread Probability Intensity",
    "slope": "Terrain Slope Angle",
    "barrier_width_m": "Cleared Barrier Width",
    "distance_to_fire": "Proximity to Active Flame Front",
    "elevation": "Surface Elevation",
    "attack_angle": "Acute Fire Attack Angle"
}

@dataclass
class SegmentPrediction:
    segment_id: str
    fire_id: str
    breach_probability: float
    hold_probability: float
    risk_tier: str
    vulnerability_rank: int
    priority_percentile: float
    priority_recommended: bool
    top_risk_factors: List[str]
    protective_factors: List[str]
    explanation: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class CLBIDecisionEngine:
    def __init__(self, model_path: Optional[str] = None):
        if model_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            model_path = os.path.join(base_dir, "models", "clbi_logistic_regression.joblib")
            
        self.model_path = model_path
        self.metadata_path = os.path.join(os.path.dirname(model_path), "clbi_model_metadata.json")
        self.features = CLEAN_FEATURES
        self.pipeline: Optional[Any] = None
        self.feature_coefficients_: Dict[str, float] = {}
        self.intercept_: float = 0.0
        self.metadata_: Dict[str, Any] = {}
        
        self.load_model(self.model_path)

    def load_model(self, model_path: str):
        """Loads a pre-trained Logistic Regression model artifact."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"CLBI model artifact not found at '{model_path}'. "
                "Please run training first: python -m src.model.train_clbi_model"
            )
            
        self.pipeline = joblib.load(model_path)
        clf = self.pipeline.named_steps['clf']
        # Breach log-odds coefficients (-coef for label=0 breach)
        breach_coefs = -clf.coef_[0]
        self.feature_coefficients_ = dict(zip(self.features, breach_coefs.tolist()))
        self.intercept_ = float(-clf.intercept_[0])
        
        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, "r") as f:
                self.metadata_ = json.load(f)

    def get_model_metadata(self) -> Dict[str, Any]:
        """Returns metadata associated with the loaded production model."""
        return self.metadata_

    def predict_df(self, df: pd.DataFrame, top_k_recommend_pct: float = 10.0) -> pd.DataFrame:
        """
        Executes CLBI inference over a DataFrame of candidate segments using the loaded model.
        Returns a DataFrame enriched with breach probability, risk tier, rank, and explanations.
        """
        self.validate_input_features(df)
        
        X = df[self.features].values
        
        # Predict probability of Held (label=1)
        p_held = self.pipeline.predict_proba(X)[:, 1]
        p_breach = 1.0 - p_held
        
        n_samples = len(df)
        
        # Rank descending by breach probability (Rank 1 = Highest breach risk)
        sorted_indices = np.argsort(-p_breach)
        vulnerability_ranks = np.empty(n_samples, dtype=int)
        vulnerability_ranks[sorted_indices] = np.arange(1, n_samples + 1)
        
        # Priority percentile: 100% = Highest vulnerability rank, 0% = Lowest vulnerability rank
        priority_percentiles = np.round((1.0 - (vulnerability_ranks - 1) / n_samples) * 100.0, 2)
        
        # Recommended inspection set cutoff
        k_count = max(1, int(n_samples * (top_k_recommend_pct / 100.0)))
        priority_recommended = vulnerability_ranks <= k_count
        
        # Assign Risk Tiers
        risk_tiers = []
        for p in p_breach:
            if p < 0.30:
                risk_tiers.append("LOW")
            elif p < 0.60:
                risk_tiers.append("MEDIUM")
            elif p < 0.80:
                risk_tiers.append("HIGH")
            else:
                risk_tiers.append("CRITICAL")
                
        # Scaled Feature Contributions for Explainability
        scaler = self.pipeline.named_steps['scaler']
        X_scaled = scaler.transform(X)
        clf = self.pipeline.named_steps['clf']
        breach_coef_vec = -clf.coef_[0]
        contributions = X_scaled * breach_coef_vec
        
        top_risk_factors_list = []
        protective_factors_list = []
        explanations_list = []
        
        for i in range(n_samples):
            contrib_row = contributions[i]
            row_dict = df.iloc[i]
            p_b = p_breach[i]
            tier = risk_tiers[i]
            
            sorted_feat_idx = np.argsort(-contrib_row)
            
            risk_feats = []
            prot_feats = []
            
            for f_idx in sorted_feat_idx:
                f_name = self.features[f_idx]
                c_val = contrib_row[f_idx]
                f_label = FEATURE_LABELS[f_name]
                f_val = row_dict[f_name]
                
                if c_val > 0.1:
                    risk_feats.append(f"{f_label} ({f_name}={f_val})")
                elif c_val < -0.1:
                    prot_feats.append(f"{f_label} ({f_name}={f_val})")
                    
            top_risk_factors_list.append(risk_feats[:3])
            protective_factors_list.append(prot_feats[:3])
            
            explanation_str = self._build_explanation(tier, p_b, risk_feats[:2], prot_feats[:2])
            explanations_list.append(explanation_str)
            
        result_df = df.copy()
        result_df["breach_probability"] = np.round(p_breach, 4)
        result_df["hold_probability"] = np.round(1.0 - p_breach, 4)
        result_df["risk_tier"] = risk_tiers
        result_df["vulnerability_rank"] = vulnerability_ranks
        result_df["priority_percentile"] = priority_percentiles
        result_df["priority_recommended"] = priority_recommended
        result_df["top_risk_factors"] = top_risk_factors_list
        result_df["protective_factors"] = protective_factors_list
        result_df["explanation"] = explanations_list
        
        return result_df

    def validate_input_features(self, df: pd.DataFrame):
        """Strict input validation for feature presence, numeric dtypes, NaN, Inf, and value bounds."""
        # 1. Column presence check
        missing_cols = [col for col in self.features if col not in df.columns]
        if missing_cols:
            raise ValueError(f"DataFrame is missing required CLBI model features: {missing_cols}")
            
        # 2. Numeric dtype check
        for col in self.features:
            if not pd.api.types.is_numeric_dtype(df[col]):
                raise ValueError(f"Feature '{col}' must be numeric, got dtype {df[col].dtype}")
                
        # 3. NaN check
        for col in self.features:
            if df[col].isnull().any():
                nan_count = df[col].isnull().sum()
                raise ValueError(f"Feature '{col}' contains {nan_count} NaN values. Missing values are not permitted.")
                
        # 4. Infinite values check
        for col in self.features:
            if np.isinf(df[col]).any():
                raise ValueError(f"Feature '{col}' contains infinite (Inf/-Inf) values.")
                
        # 5. Value bounds checks based on project specs
        if (df["burn_prob"] < 0.0).any() or (df["burn_prob"] > 1.0).any():
            raise ValueError("Feature 'burn_prob' must be bounded in [0.0, 1.0]")
            
        if (df["attack_dot_product"] < -1.0).any() or (df["attack_dot_product"] > 1.0).any():
            raise ValueError("Feature 'attack_dot_product' must be bounded in [-1.0, 1.0]")
            
        if (df["slope"] < 0.0).any():
            raise ValueError("Feature 'slope' cannot be negative")
            
        if (df["distance_to_fire"] < 0.0).any():
            raise ValueError("Feature 'distance_to_fire' cannot be negative")
            
        if (df["barrier_width_m"] <= 0.0).any():
            raise ValueError("Feature 'barrier_width_m' must be positive (> 0)")

    def prioritize_candidate_line(
        self,
        df_predicted: pd.DataFrame,
        resource_pct: Optional[float] = None,
        resource_count: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Risk-based candidate-line prioritization function.
        Selects top vulnerable segments based on descending breach probability.
        """
        n_total = len(df_predicted)
        if resource_count is not None:
            k_select = max(1, min(n_total, resource_count))
        elif resource_pct is not None:
            k_select = max(1, min(n_total, int(n_total * (resource_pct / 100.0))))
        else:
            k_select = max(1, int(n_total * 0.10))
            
        sorted_df = df_predicted.sort_values(by="vulnerability_rank", ascending=True).reset_index(drop=True)
        selected_df = sorted_df.iloc[:k_select].copy()
        
        return {
            "selected_segments": selected_df,
            "total_candidate_segments": n_total,
            "total_selected_segments": k_select,
            "pct_line_selected": round((k_select / n_total) * 100, 2),
            "selected_mean_breach_probability": round(float(selected_df["breach_probability"].mean()), 4),
            "selected_risk_tier_counts": selected_df["risk_tier"].value_counts().to_dict(),
            "prioritization_type": "risk-based candidate-line prioritization"
        }

    def summarize_risk(self, df_predicted: pd.DataFrame) -> Dict[str, Any]:
        """Calculates risk tier counts and summary statistics across all candidate segments."""
        n_total = len(df_predicted)
        tier_counts = df_predicted["risk_tier"].value_counts().to_dict()
        
        for tier in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]:
            tier_counts.setdefault(tier, 0)
            
        high_crit_count = tier_counts["HIGH"] + tier_counts["CRITICAL"]
        high_crit_pct = (high_crit_count / n_total) * 100.0 if n_total > 0 else 0.0
        
        top_5_count  = max(1, int(n_total * 0.05))
        top_10_count = max(1, int(n_total * 0.10))
        top_20_count = max(1, int(n_total * 0.20))
        
        return {
            "total_segments": n_total,
            "risk_tier_counts": tier_counts,
            "mean_breach_probability": round(float(df_predicted["breach_probability"].mean()), 4),
            "max_breach_probability": round(float(df_predicted["breach_probability"].max()), 4),
            "min_breach_probability": round(float(df_predicted["breach_probability"].min()), 4),
            "high_or_critical_count": high_crit_count,
            "pct_line_high_or_critical": round(high_crit_pct, 2),
            "top_5_pct_cutoff_count": top_5_count,
            "top_10_pct_cutoff_count": top_10_count,
            "top_20_pct_cutoff_count": top_20_count
        }

    def _build_explanation(
        self,
        tier: str,
        p_breach: float,
        top_risks: List[str],
        top_prots: List[str]
    ) -> str:
        p_pct = p_breach * 100.0
        if tier == "CRITICAL":
            desc = f"CRITICAL breach vulnerability ({p_pct:.1f}% predicted risk)."
        elif tier == "HIGH":
            desc = f"HIGH breach risk ({p_pct:.1f}% predicted risk)."
        elif tier == "MEDIUM":
            desc = f"MEDIUM breach risk ({p_pct:.1f}% predicted risk)."
        else:
            desc = f"LOW breach risk ({p_pct:.1f}% predicted risk)."
            
        parts = [desc]
        if top_risks:
            risk_str = ", ".join(top_risks)
            parts.append(f"Elevated predicted risk is associated with: {risk_str}.")
        if top_prots:
            prot_str = ", ".join(top_prots)
            parts.append(f"Holding probability is supported by: {prot_str}.")
            
        return " ".join(parts)
