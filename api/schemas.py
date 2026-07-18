"""Pydantic request/response schemas for the prediction API."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class PatientFeatures(BaseModel):
    """Clinical features used by the Heart Disease UCI classifier."""

    age: float = Field(..., ge=1, le=120, description="Age in years", example=63)
    sex: int = Field(..., ge=0, le=1, description="Sex (1 = male, 0 = female)", example=1)
    cp: int = Field(..., ge=0, le=3, description="Chest pain type (0-3)", example=3)
    trestbps: float = Field(
        ..., ge=50, le=250, description="Resting blood pressure (mm Hg)", example=145
    )
    chol: float = Field(
        ..., ge=50, le=600, description="Serum cholesterol (mg/dl)", example=233
    )
    fbs: int = Field(
        ..., ge=0, le=1, description="Fasting blood sugar > 120 mg/dl (1/0)", example=1
    )
    restecg: int = Field(
        ..., ge=0, le=2, description="Resting ECG results (0-2)", example=0
    )
    thalach: float = Field(
        ..., ge=50, le=250, description="Maximum heart rate achieved", example=150
    )
    exang: int = Field(
        ..., ge=0, le=1, description="Exercise induced angina (1/0)", example=0
    )
    oldpeak: float = Field(
        ..., ge=0, le=10, description="ST depression induced by exercise", example=2.3
    )
    slope: int = Field(
        ..., ge=0, le=2, description="Slope of peak exercise ST segment", example=0
    )
    ca: int = Field(
        ..., ge=0, le=4, description="Number of major vessels colored by fluoroscopy", example=0
    )
    thal: int = Field(
        ...,
        ge=0,
        le=3,
        description="Thalassemia (0=unknown/normal variants depending on encoding)",
        example=1,
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "age": 63,
                "sex": 1,
                "cp": 3,
                "trestbps": 145,
                "chol": 233,
                "fbs": 1,
                "restecg": 0,
                "thalach": 150,
                "exang": 0,
                "oldpeak": 2.3,
                "slope": 0,
                "ca": 0,
                "thal": 1,
            }
        }
    }


class PredictionResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    prediction: int = Field(..., description="0 = low risk / no disease, 1 = disease risk")
    prediction_label: str
    confidence: float = Field(..., ge=0, le=1, description="Probability of predicted class")
    probability_disease: float = Field(
        ..., ge=0, le=1, description="Probability of heart disease (class 1)"
    )
    model_name: Optional[str] = None


class HealthResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    status: str
    model_loaded: bool
    model_name: Optional[str] = None


class BatchPredictionRequest(BaseModel):
    patients: List[PatientFeatures]
