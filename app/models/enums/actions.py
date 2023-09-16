import enum


class ActionEnum(enum.Enum):
    """
    defining the enum to recognize the actions mentioned in the input text
    """
    SEARCH = 0
    SEND = 1
    DOWNLOAD = 2
