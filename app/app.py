from fastapi import FastAPI

from models.documents import ActionAndText, RecognizeActionInput

app = FastAPI()


@app.post("/recognize_action/")
def read_root(text: RecognizeActionInput) -> ActionAndText:
    return ActionAndText(
        text=text.query,
        action=""
    )
