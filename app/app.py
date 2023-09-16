from fastapi import FastAPI

from libs.recognize_action import recognize_action
from libs.transformers_ import load_model_and_tokenizer
from models.documents import ActionAndText, RecognizeActionInput

app = FastAPI()


@app.post("/recognize_action/")
def read_root(text: RecognizeActionInput) -> ActionAndText:
    # todo - load model on app starting
    model, tokenizer = load_model_and_tokenizer('thameemk/actions-recognizer', 3)
    res = recognize_action(model, tokenizer, text.text)
    return res
