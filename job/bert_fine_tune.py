import evaluate
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import TrainingArguments, Trainer

from libs.transformers_ import load_model_and_tokenizer


# noinspection PyUnresolvedReferences
class Dataset(torch.utils.data.Dataset):
    """
    # creating torch dataset
    """

    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels:
            item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])


def compute_metrics(eval_pred):
    metric = evaluate.load("accuracy")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


def splitting_and_tokenize_data(x, y, tokenizer):
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, stratify=y)
    x_train_tokenized = tokenizer(x_train, padding=True, truncation=True, max_length=512)
    x_val_tokenized = tokenizer(x_val, padding=True, truncation=True, max_length=512)

    return x_train_tokenized, x_val_tokenized, y_train, y_val


def bert_fine_tune():
    data = pd.read_csv('data/dataset_v3.csv')
    model, tokenizer = load_model_and_tokenizer('bert-base-uncased', 3)

    x_train_tokenized, x_val_tokenized, y_train, y_val = splitting_and_tokenize_data(list(data["text"]),
                                                                                     list(data["label"]), tokenizer)

    train_dataset = Dataset(x_train_tokenized, y_train)
    validation_dataset = Dataset(x_val_tokenized, y_val)

    training_args = TrainingArguments(output_dir="test_trainer/", evaluation_strategy="epoch", num_train_epochs=3,
                                      logging_dir='logs/', )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    trainer.evaluate()

    trainer.save_model('actions-recognizer')
    tokenizer.save_pretrained('actions-recognizer')


if __name__ == '__main__':
    bert_fine_tune()
