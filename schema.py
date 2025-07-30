from pydantic import BaseModel
from typing import Optional, Dict

class GraphMessage(BaseModel):
    role: str            # "user" | "assistant" | "delegate"
    text: Optional[str]
    data: Optional[Dict] = None
