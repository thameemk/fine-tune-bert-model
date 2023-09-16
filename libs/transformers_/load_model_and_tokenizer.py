from transformers import AutoTokenizer, AutoModelForSequenceClassification

from settings import HUGGING_FACE_ACCESS_TOKEN


def load_model_and_tokenizer(model_name: str, num_labels: int):
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels,
                                                               token=HUGGING_FACE_ACCESS_TOKEN)
    tokenizer = AutoTokenizer.from_pretrained(model_name, num_labels=num_labels, token=HUGGING_FACE_ACCESS_TOKEN)

    return model, tokenizer
