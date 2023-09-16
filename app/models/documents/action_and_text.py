from pydantic import BaseModel

from app.models.enums import ActionEnum


class ActionAndText(BaseModel):
    text: str
    action: ActionEnum
