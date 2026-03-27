from pydantic import BaseModel
from typing import Optional


class Respondent(BaseModel):
    respondent_id: str
    age_band: Optional[str] = None
    gender: Optional[str] = None
    region: Optional[str] = None
    segment: Optional[str] = None
    category_usage_freq: Optional[str] = None
    weight: float = 1.0
