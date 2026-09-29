"""
schemas.py - Pydantic Request & Response Schemas for CLBI REST API
"""

import math
from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field, field_validator, model_validator

class SegmentInput(BaseModel):
    segment_id: str = Field(default="seg_001", description="Unique identifier for the segment")
    fire_id: Optional[str] = Field(default=None, description="Wildfire incident ID")
    fire_name: Optional[str] = Field(default=None, description="Wildfire incident name")
    date: Optional[str] = Field(default=None, description="Observation date (YYYY-MM-DD)")
    
    slope: float = Field(..., description="Terrain slope angle in degrees (>= 0)")
    elevation: float = Field(..., description="Surface elevation in meters")
    distance_to_fire: float = Field(..., description="Distance to active flame front in meters (>= 0)")
    barrier_width_m: float = Field(..., description="Cleared barrier width in meters (> 0)")
    burn_prob: float = Field(..., description="Predicted next-day burn probability intensity in [0.0, 1.0]")
    attack_angle: float = Field(..., description="Acute fire attack interaction angle in degrees")
    attack_dot_product: float = Field(..., description="Direct head-fire attack alignment ratio in [-1.0, 1.0]")

    @field_validator("slope", "elevation", "distance_to_fire", "barrier_width_m", "burn_prob", "attack_angle", "attack_dot_product")
    @classmethod
    def check_non_nan_inf(cls, value: float, info) -> float:
        if math.isnan(value) or math.isinf(value):
            raise ValueError(f"Feature '{info.field_name}' must be a finite numeric value (got {value})")
        return value

    @field_validator("burn_prob")
    @classmethod
    def check_burn_prob_range(cls, value: float) -> float:
        if value < 0.0 or value > 1.0:
            raise ValueError("burn_prob must be between 0 and 1")
        return value

    @field_validator("attack_dot_product")
    @classmethod
    def check_attack_dot_product_range(cls, value: float) -> float:
        if value < -1.0 or value > 1.0:
            raise ValueError("attack_dot_product must be between -1 and 1")
        return value

    @field_validator("slope")
    @classmethod
    def check_slope(cls, value: float) -> float:
        if value < 0.0:
            raise ValueError("slope cannot be negative")
        return value

    @field_validator("distance_to_fire")
    @classmethod
    def check_distance_to_fire(cls, value: float) -> float:
        if value < 0.0:
            raise ValueError("distance_to_fire cannot be negative")
        return value

    @field_validator("barrier_width_m")
    @classmethod
    def check_barrier_width(cls, value: float) -> float:
        if value <= 0.0:
            raise ValueError("barrier_width_m must be positive (> 0)")
        return value


class SingleSegmentPredictionResponse(BaseModel):
    segment_id: str
    fire_id: Optional[str] = None
    fire_name: Optional[str] = None
    date: Optional[str] = None
    breach_probability: float
    hold_probability: float
    risk_tier: str
    vulnerability_rank: Optional[int] = None
    priority_percentile: Optional[float] = None
    priority_recommended: Optional[bool] = None
    top_risk_factors: List[str]
    protective_factors: List[str]
    explanation: str


class BatchSegmentPredictionResponse(BaseModel):
    segment_id: str
    fire_id: Optional[str] = None
    fire_name: Optional[str] = None
    date: Optional[str] = None
    breach_probability: float
    hold_probability: float
    risk_tier: str
    vulnerability_rank: int
    priority_percentile: float
    priority_recommended: bool
    top_risk_factors: List[str]
    protective_factors: List[str]
    explanation: str


class BatchPredictRequest(BaseModel):
    segments: List[SegmentInput] = Field(..., description="List of candidate line segments for inference")

    @field_validator("segments")
    @classmethod
    def check_non_empty(cls, value: List[SegmentInput]) -> List[SegmentInput]:
        if not value:
            raise ValueError("Segment list cannot be empty")
        return value


class BatchPredictResponse(BaseModel):
    total_segments: int
    segments: List[BatchSegmentPredictionResponse]


class SelectionCriteria(BaseModel):
    type: Literal["percentage", "count"] = Field(..., description="Selection type: 'percentage' or 'count'")
    value: float = Field(..., description="Selection numeric value")

    @model_validator(mode="after")
    def validate_selection_bounds(self) -> "SelectionCriteria":
        if self.type == "percentage":
            if self.value <= 0.0 or self.value > 100.0:
                raise ValueError("selection percentage value must be between 0 and 100 (exclusive of 0)")
        elif self.type == "count":
            if self.value < 1.0 or int(self.value) != self.value:
                raise ValueError("selection count value must be an integer >= 1")
        return self


class PrioritizationRequest(BaseModel):
    segments: List[SegmentInput] = Field(..., description="List of candidate segments to prioritize")
    selection: SelectionCriteria = Field(..., description="Prioritization cutoff selection criteria")

    @field_validator("segments")
    @classmethod
    def check_non_empty(cls, value: List[SegmentInput]) -> List[SegmentInput]:
        if not value:
            raise ValueError("Segment list cannot be empty")
        return value


class PrioritizationResponse(BaseModel):
    total_segments: int
    selected_count: int
    selection_type: str
    selection_value: float
    segments: List[BatchSegmentPredictionResponse]


class HealthResponse(BaseModel):
    status: str = "ok"
    service: str = "CLBI API"
    version: str = "1.0.0"


class ModelInfoResponse(BaseModel):
    model_type: str
    feature_list: List[str]
    risk_thresholds: Dict[str, float]
    training_rows: int
    training_fires: List[str]
    evaluation: Dict[str, Any]


class ErrorResponse(BaseModel):
    error: str
    message: str
