from transformers import AutoTokenizer, AutoModelForSequenceClassification


def load_model_and_tokenizer(model_name: str, num_labels: int):
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    tokenizer = AutoTokenizer.from_pretrained(model_name, num_labels=num_labels)

    return model, tokenizer
