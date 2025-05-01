from pydantic import BaseModel, Field, field_validator
from typing import List
import re

class PredictRequest(BaseModel):
  text: str = Field(
    ...,
    example="Cows lose their jobs as milk prices drop",
    min_lenth=10,
    max_length=500,
  )

  @field_validator("text", mode="before")
  def clean_text(cls, v: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9\s\.,:;!?'\"-]+", "", v)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned

class LabelScore(BaseModel):
  code:  int = Field(..., description="Numeric label ID")
  label: str = Field(..., description="Human-readable label")
  score: float = Field(..., description="Probability")

class PredictResponse(BaseModel):
  model: str = Field(..., description="Driver used for inference")
  category: LabelScore = Field(..., description="Highest-confidence prediction")

class Top5Response(BaseModel):
  model: str
  top5:  List[LabelScore]
