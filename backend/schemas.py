from pydantic import BaseModel, Field
from typing import List, Dict, Optional

class PredictIn(BaseModel):
    text: str
    k: int = 3
    limit: int = 3
    mode: str = "supervised"

class MoodScore(BaseModel):
    label: str
    score: float

class AnalyzeOut(BaseModel):
    moods: List[MoodScore]
    tracks: List[Dict]
    meta: Dict = {}   # include uncertainty + top2 gap so UI can nudge

class FeedbackIn(BaseModel):
    text: str
    chosen: List[str] = Field(default_factory=list)
    rejected: List[str] = Field(default_factory=list)
    user_id_hash: Optional[str] = None
