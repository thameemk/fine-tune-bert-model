from pydantic import BaseModel


class RecognizeActionInput(BaseModel):
    text: str
