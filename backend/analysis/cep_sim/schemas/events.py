from pydantic import BaseModel
from typing import Optional, Dict, Any


class EpisodicEvent(BaseModel):
    event_id: str
    respondent_id: str
    event_type: str
    brand_id: Optional[str] = None
    cep_id: Optional[str] = None
    event_time: str
    context_json: Dict[str, Any]
    strength: float
    source: str
