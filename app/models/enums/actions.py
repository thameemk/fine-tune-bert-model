import enum


class ActionEnum(enum.Enum):
    """
    defining the enum to recognize the actions mentioned in the input text
    """
    SEARCH = "SEARCH"
    SEND = "SEND"
    DOWNLOAD = "DOWNLOAD"
