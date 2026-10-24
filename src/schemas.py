from pydantic import BaseModel

class ResearchAnswer(BaseModel):
    answer: str
    key_points: list[str]
    sources: list[str]