from transformers import AutoTokenizer, BertForQuestionAnswering
import torch

from src.data.build_dataset import get_content_from_wikipedia

"""
#todo

custom dataset
training and testing [8:2]

Tokenize and format input query
Fine-tune BERT
Predict action
perform the action
"""


class ModelAndTokenizer:
    """
    load the model and tokenizer.
    """

    def __init__(self):
        self.model = None
        self.tokenizer = None

    def load_model_and_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained("deepset/bert-base-cased-squad2")
        self.model = BertForQuestionAnswering.from_pretrained("deepset/bert-base-cased-squad2")


def query_result(query: str, context: str, tokenizer, model):
    inputs = tokenizer(query, context, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    answer_start_index = outputs.start_logits.argmax()
    answer_end_index = outputs.end_logits.argmax()

    predict_answer_tokens = inputs.input_ids[0, answer_start_index: answer_end_index + 1]
    answer = tokenizer.decode(predict_answer_tokens, skip_special_tokens=True)

    return answer


if __name__ == '__main__':
    cus_model = ModelAndTokenizer()
    cus_model.load_model_and_tokenizer()
    print(query_result(query="search for quantum mechanics", context=get_content_from_wikipedia("Quantum mechanics"),
                       model=cus_model.model, tokenizer=cus_model.tokenizer))
