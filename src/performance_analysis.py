import enum

from transformers import AutoTokenizer, AutoModelForSequenceClassification


class ActionEnum(enum.Enum):
    SEARCH = 0
    SEND = 1
    DOWNLOAD = 2


def performance_analysis():
    model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', num_labels=3)

    model_v2 = AutoModelForSequenceClassification.from_pretrained('search_action', num_labels=3)
    tokenizer_v2 = AutoTokenizer.from_pretrained('search_action', num_labels=3)

    def compare_results(input_text):
        inputs = tokenizer(input_text, return_tensors="pt")
        predicted_class_id = model(**inputs).logits.argmax().item()

        inputs_v2 = tokenizer_v2(input_text, return_tensors="pt")
        predicted_class_id_v2 = model_v2(**inputs_v2).logits.argmax().item()

        return {
            'input_text': input_text,
            'pre_trained': ActionEnum(predicted_class_id).name,
            'fine_tuned': ActionEnum(predicted_class_id_v2).name
        }

    return [compare_results("send the documentation to the team"), compare_results("download the game from store"),
            compare_results("seach for healthy food"), compare_results("download open source github projects")]


if __name__ == '__main__':
    performance_analysis()
