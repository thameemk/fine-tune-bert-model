from models.documents import ActionAndText
from models.enums import ActionEnum


def recognize_action(model, tokenizer, text: str) -> ActionAndText:
    """

    Args:
        model: the large language / fine-tuned model used
        tokenizer: the tokenizer used to encode the input
        text:the input text in which we need recognize the action

    Returns:
        The function return the recognized action along with the given text

    """
    inputs = tokenizer(text, return_tensors="pt")
    predicted_class_id = model(**inputs).logits.argmax().item()

    return ActionAndText(
        text=text,
        action=ActionEnum(predicted_class_id)
    )
