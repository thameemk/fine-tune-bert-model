import re


def text_preprocessing(raw_text: str):
    """
    used to clean the raw data
    """

    raw_text = raw_text.lower()
    raw_text = re.sub(r"what's", "what is ", raw_text)
    raw_text = re.sub(r"won't", "will not ", raw_text)
    raw_text = re.sub(r"\'s", " ", raw_text)
    raw_text = re.sub(r"\'ve", " have ", raw_text)
    raw_text = re.sub(r"can't", "can not ", raw_text)
    raw_text = re.sub(r"n't", " not ", raw_text)
    raw_text = re.sub(r"i'm", "i am ", raw_text)
    raw_text = re.sub(r"\'re", " are ", raw_text)
    raw_text = re.sub(r"\'d", " would ", raw_text)
    raw_text = re.sub(r"\'ll", " will ", raw_text)
    raw_text = re.sub(r"\'scuse", " excuse ", raw_text)
    raw_text = re.sub(r"\'\n", " ", raw_text)
    raw_text = re.sub(r"-", " ", raw_text)
    raw_text = re.sub(r"\'\xa0", " ", raw_text)
    raw_text = re.sub('\s+', ' ', raw_text)
    raw_text = ''.join(c for c in raw_text if not c.isnumeric())

    # Remove '@name'
    raw_text = re.sub(r'(@.*?)[\s]', ' ', raw_text)

    # Replace '&amp;' with '&'
    raw_text = re.sub(r'&amp;', '&', raw_text)

    # Remove trailing whitespace
    raw_text = re.sub(r'\s+', ' ', raw_text).strip()

    return raw_text
