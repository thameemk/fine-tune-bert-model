import os

from libs.transformers_ import load_model_and_tokenizer

HUGGING_FACE_ACCESS_TOKEN = os.environ.get('HUGGING_FACE_ACCESS_TOKEN')


def push_to_huggingface_hub():
    model, tokenizer = load_model_and_tokenizer('actions-recognizer', 3)
    model.push_to_hub("actions-recognizer", token=HUGGING_FACE_ACCESS_TOKEN)
    tokenizer.push_to_hub("actions-recognizer", token=HUGGING_FACE_ACCESS_TOKEN)


if __name__ == '__main__':
    push_to_huggingface_hub()
