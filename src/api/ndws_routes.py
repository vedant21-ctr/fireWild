"""
ndws_routes.py - FastAPI Route Handlers for Next Day Wildfire Spread (NDWS) Benchmarks

Exposes sanitized model evaluation metrics and illustrative spatial demonstration
patches for frontend consumption in accordance with docs/PHASE1_UI_INTEGRATION_CONTRACT.md.
"""

from typing import Dict, Any
from fastapi import APIRouter, HTTPException, status

from src.api.ndws_contracts import (
    get_sanitized_benchmark_payload,
    generate_illustrative_spatial_demo_patch,
)

ndws_router = APIRouter(prefix="/api/v1/ndws", tags=["NDWS Tabular Benchmark API"])


@ndws_router.get(
    "/benchmark/metrics",
    summary="NDWS Phase 1 Tabular Benchmark Metrics",
    response_description="Sanitized performance metrics and model comparisons across baselines and LightGBM.",
)
def get_benchmark_metrics() -> Dict[str, Any]:
    """
    Returns sanitized comparative benchmark metrics (AP, ROC-AUC, Brier score,
    F1, precision, recall, confusion matrix counts) for all Phase 1 tabular models.
    All private machine filesystem paths are stripped.
    """
    try:
        return get_sanitized_benchmark_payload()
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"error": "artifact_missing", "message": str(e)},
        )


@ndws_router.get(
    "/demo/patch",
    summary="Illustrative 64x64 Spatial Spread Demo Patch",
    response_description="Demonstration 64x64 grid of spread probabilities for UI map/heatmap preview.",
)
def get_spatial_demo_patch() -> Dict[str, Any]:
    """
    Returns a deterministic 64x64 spread probability matrix explicitly marked
    as an illustrative mockup for UI heatmap and layout demonstration.
    """
    return generate_illustrative_spatial_demo_patch()
