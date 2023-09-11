from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split

import pandas as pd


def load_data():
    data = pd.read_csv("data/dataset.csv")
    return data


def get_model_and_tokenizer():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

    return model, tokenizer


def data_splitting(x, y, tokenizer):
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, stratify=y)
    x_train_tokenized = tokenizer(x_train, padding=True, truncation=True, max_length=512)
    x_val_tokenized = tokenizer(x_val, padding=True, truncation=True, max_length=512)

    return x_train_tokenized, x_val_tokenized
