"""
routes.py - API Route Handlers for Candidate Line Breach Intelligence (CLBI)
"""

import os
import pandas as pd
from typing import Dict, Any, List
from fastapi import APIRouter, HTTPException, Depends

from src.model.clbi_model import CLBIDecisionEngine
from src.api.schemas import (
    SegmentInput,
    SingleSegmentPredictionResponse,
    BatchSegmentPredictionResponse,
    BatchPredictRequest,
    BatchPredictResponse,
    PrioritizationRequest,
    PrioritizationResponse,
    HealthResponse,
    ModelInfoResponse
)

router = APIRouter(prefix="/api/v1", tags=["CLBI API"])

# Global singleton or dependency for decision engine
_ENGINE_INSTANCE: float = None

def get_engine() -> CLBIDecisionEngine:
    try:
        return CLBIDecisionEngine()
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "model_artifact_missing",
                "message": str(e)
            }
        )

@router.get("/health", response_model=HealthResponse, summary="System Health Check")
def get_health() -> HealthResponse:
    """Returns service health status and API version."""
    return HealthResponse(status="ok", service="CLBI API", version="1.0.0")

@router.get("/model/info", response_model=ModelInfoResponse, summary="Model Provenance & Info")
def get_model_info(engine: CLBIDecisionEngine = Depends(get_engine)) -> ModelInfoResponse:
    """Returns metadata, feature list, risk thresholds, and evaluation metrics for the production model."""
    metadata = engine.get_model_metadata()
    return ModelInfoResponse(
        model_type=metadata.get("model_type", "Standardized 7-Feature Logistic Regression Pipeline"),
        feature_list=engine.features,
        risk_thresholds={
            "low": 0.30,
            "medium": 0.60,
            "high": 0.80
        },
        training_rows=metadata.get("training_rows", 3050),
        training_fires=metadata.get("training_fires", ["CA-MNF-013028", "CA-SHU-007808", "CA-SNF-000958"]),
        evaluation=metadata.get("evaluation", {
            "method": "5-fold Leave-One-Fire-Out",
            "mean_pr_auc": 0.9508,
            "brier": 0.1199,
            "ece_10": 0.0109,
            "ece_20": 0.0151
        })
    )

@router.post("/predict", response_model=SingleSegmentPredictionResponse, summary="Single Segment Risk Prediction")
def predict_single_segment(
    segment: SegmentInput,
    engine: CLBIDecisionEngine = Depends(get_engine)
) -> SingleSegmentPredictionResponse:
    """
    Runs breach probability prediction and risk stratification for a single candidate fireline segment.
    Population rank fields (vulnerability_rank, priority_percentile, priority_recommended) are set to null.
    """
    df_input = pd.DataFrame([segment.model_dump()])
    try:
        predicted_df = engine.predict_df(df_input)
    except ValueError as val_err:
        raise HTTPException(status_code=400, detail={"error": "validation_error", "message": str(val_err)})

    row = predicted_df.iloc[0]
    return SingleSegmentPredictionResponse(
        segment_id=row["segment_id"],
        fire_id=row.get("fire_id"),
        fire_name=row.get("fire_name"),
        date=str(row["date"]) if pd.notnull(row.get("date")) else None,
        breach_probability=float(row["breach_probability"]),
        hold_probability=float(row["hold_probability"]),
        risk_tier=row["risk_tier"],
        vulnerability_rank=None,
        priority_percentile=None,
        priority_recommended=None,
        top_risk_factors=row["top_risk_factors"],
        protective_factors=row["protective_factors"],
        explanation=row["explanation"]
    )

@router.post("/predict/batch", response_model=BatchPredictResponse, summary="Batch Segments Risk Prediction")
def predict_batch_segments(
    request: BatchPredictRequest,
    engine: CLBIDecisionEngine = Depends(get_engine)
) -> BatchPredictResponse:
    """
    Runs breach probability prediction, population vulnerability ranking, and risk stratification
    across a list of candidate fireline segments.
    """
    df_input = pd.DataFrame([seg.model_dump() for seg in request.segments])
    try:
        predicted_df = engine.predict_df(df_input)
    except ValueError as val_err:
        raise HTTPException(status_code=400, detail={"error": "validation_error", "message": str(val_err)})

    response_segments = []
    for idx, row in predicted_df.iterrows():
        response_segments.append(
            BatchSegmentPredictionResponse(
                segment_id=row["segment_id"],
                fire_id=row.get("fire_id"),
                fire_name=row.get("fire_name"),
                date=str(row["date"]) if pd.notnull(row.get("date")) else None,
                breach_probability=float(row["breach_probability"]),
                hold_probability=float(row["hold_probability"]),
                risk_tier=row["risk_tier"],
                vulnerability_rank=int(row["vulnerability_rank"]),
                priority_percentile=float(row["priority_percentile"]),
                priority_recommended=bool(row["priority_recommended"]),
                top_risk_factors=row["top_risk_factors"],
                protective_factors=row["protective_factors"],
                explanation=row["explanation"]
            )
        )

    return BatchPredictResponse(
        total_segments=len(response_segments),
        segments=response_segments
    )

@router.post("/prioritize", response_model=PrioritizationResponse, summary="Candidate Line Prioritization")
def prioritize_segments(
    request: PrioritizationRequest,
    engine: CLBIDecisionEngine = Depends(get_engine)
) -> PrioritizationResponse:
    """
    Ranks candidate segments by breach probability (descending) and selects the top-k segments
    based on specified percentage or count selection criteria.
    """
    df_input = pd.DataFrame([seg.model_dump() for seg in request.segments])
    try:
        predicted_df = engine.predict_df(df_input)
    except ValueError as val_err:
        raise HTTPException(status_code=400, detail={"error": "validation_error", "message": str(val_err)})

    sel_type = request.selection.type
    sel_val = request.selection.value

    if sel_type == "percentage":
        res = engine.prioritize_candidate_line(predicted_df, resource_pct=sel_val)
    else:
        res = engine.prioritize_candidate_line(predicted_df, resource_count=int(sel_val))

    selected_df = res["selected_segments"]
    selected_count = res["total_selected_segments"]

    response_segments = []
    for idx, row in selected_df.iterrows():
        response_segments.append(
            BatchSegmentPredictionResponse(
                segment_id=row["segment_id"],
                fire_id=row.get("fire_id"),
                fire_name=row.get("fire_name"),
                date=str(row["date"]) if pd.notnull(row.get("date")) else None,
                breach_probability=float(row["breach_probability"]),
                hold_probability=float(row["hold_probability"]),
                risk_tier=row["risk_tier"],
                vulnerability_rank=int(row["vulnerability_rank"]),
                priority_percentile=float(row["priority_percentile"]),
                priority_recommended=bool(row["priority_recommended"]),
                top_risk_factors=row["top_risk_factors"],
                protective_factors=row["protective_factors"],
                explanation=row["explanation"]
            )
        )

    return PrioritizationResponse(
        total_segments=len(request.segments),
        selected_count=selected_count,
        selection_type=sel_type,
        selection_value=sel_val,
        segments=response_segments
    )

@router.get("/demo", summary="CLBI Demonstration Endpoint")
def get_demo_response(engine: CLBIDecisionEngine = Depends(get_engine)) -> Dict[str, Any]:
    """
    Generates and returns product-ready 100-segment demo predictions response matching
    the canonical benchmark demonstration format.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    demo_csv_path = os.path.join(base_dir, "data", "demo", "clbi_demo_segments.csv")

    if not os.path.exists(demo_csv_path):
        from src.model.clbi_demo import create_demo_dataset
        demo_df = create_demo_dataset()
    else:
        demo_df = pd.read_csv(demo_csv_path)

    predicted_df = engine.predict_df(demo_df, top_k_recommend_pct=10.0)
    summary = engine.summarize_risk(predicted_df)

    segment_json_list = []
    for idx, row in predicted_df.iterrows():
        segment_json_list.append({
            "segment_id": str(row["segment_id"]),
            "fire_id": str(row["fire_id"]),
            "fire_name": str(row["fire_name"]),
            "date": str(row["date"]),
            "breach_probability": float(row["breach_probability"]),
            "hold_probability": float(row["hold_probability"]),
            "risk_tier": str(row["risk_tier"]),
            "vulnerability_rank": int(row["vulnerability_rank"]),
            "priority_percentile": float(row["priority_percentile"]),
            "priority_recommended": bool(row["priority_recommended"]),
            "top_risk_factors": row["top_risk_factors"] if isinstance(row["top_risk_factors"], list) else [],
            "protective_factors": row["protective_factors"] if isinstance(row["protective_factors"], list) else [],
            "explanation": str(row["explanation"])
        })

    return {
        "metadata": {
            "product": "Candidate Line Breach Intelligence",
            "version": "1.0.0",
            "model": "Standardized 7-Feature Logistic Regression Pipeline",
            "data_source": "NDWS-derived historical benchmark (CZU Lightning Complex)",
            "retrospective": True
        },
        "summary": {
            "total_segments": summary["total_segments"],
            "low": summary["risk_tier_counts"]["LOW"],
            "medium": summary["risk_tier_counts"]["MEDIUM"],
            "high": summary["risk_tier_counts"]["HIGH"],
            "critical": summary["risk_tier_counts"]["CRITICAL"],
            "high_or_critical": summary["high_or_critical_count"]
        },
        "prioritization": {
            "top_10_percent_count": summary["top_10_pct_cutoff_count"],
            "top_20_percent_count": summary["top_20_pct_cutoff_count"]
        },
        "segments": segment_json_list
    }
