from pydantic import BaseModel

from models.enums import ActionEnum


class ActionAndText(BaseModel):
    text: str
    action: ActionEnum
