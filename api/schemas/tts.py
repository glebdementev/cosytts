from pydantic import BaseModel, Field


class StreamingTtsRequest(BaseModel):
    text: str = Field(..., min_length=1)
    voice: str = Field(..., min_length=1)
