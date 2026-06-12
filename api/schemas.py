from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class WaterPotabilityFeatures(BaseModel):
    ph: Optional[float] = Field(None, ge=0, le=14, description="pH wody")
    Hardness: float = Field(..., ge=0, le=500, description="Twardość wody")
    Solids: float = Field(..., ge=0, le=70000, description="Całkowite rozpuszczone substancje stałe")
    Chloramines: float = Field(..., ge=0, le=20, description="Chloraminy")
    Sulfate: Optional[float] = Field(None, ge=0, le=500, description="Siarczany")
    Conductivity: float = Field(..., ge=0, le=800, description="Przewodność")
    Organic_carbon: float = Field(..., ge=0, le=30, description="Organiczny węgiel")
    Trihalomethanes: Optional[float] = Field(
        None, ge=0, le=130, description="Trihalometany"
    )
    Turbidity: float = Field(..., ge=0, le=10, description="Mętność")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "ph": 7.08,
                "Hardness": 196.37,
                "Solids": 22014.09,
                "Chloramines": 7.12,
                "Sulfate": 333.78,
                "Conductivity": 426.21,
                "Organic_carbon": 14.28,
                "Trihalomethanes": 66.40,
                "Turbidity": 3.97,
            }
        }
    )


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: Optional[str] = None
    model_type: Optional[str] = None


class PredictResponse(BaseModel):
    prediction: float
    model: str
