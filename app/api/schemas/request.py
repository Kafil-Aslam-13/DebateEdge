"""API Request schemas """
from pydantic import BaseModel , Field

class StartDebateRequest(BaseModel):
    topic: str = Field(...,min_length=10,max_length=300)
    user_side: str = Field(..., pattern="^(for|against)$")

    model_config = {
        "json_schema_extra": {
            "example": {
                "topic":     "Social media does more harm than good",
                "user_side": "for",
            }
        }
    }

class ArgueRequest(BaseModel):
    topic:        str = Field(..., min_length=10, max_length=300)
    user_side:    str = Field(..., pattern="^(for|against)$")
    argument:     str = Field(..., min_length=10, max_length=2000)
    turn_number:  int = Field(..., ge=1, le=20)

    model_config = {
        "json_schema_extra": {
            "example": {
                "topic":       "Social media does more harm than good",
                "user_side":   "for",
                "argument":    "Studies show social media increases anxiety.",
                "turn_number": 1,
            }
        }
    }