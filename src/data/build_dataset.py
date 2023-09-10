# Data Source

# Sample dataset (text) generator using wikipedia


import wikipedia

from src.data.text_preprocessing import text_preprocessing


def get_content_from_wikipedia(title: str) -> str:
    raw_content = wikipedia.summary(title)
    return text_preprocessing(raw_content)
