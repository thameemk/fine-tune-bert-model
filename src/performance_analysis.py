from libs.recognize_action import recognize_action
from libs.transformers_ import load_model_and_tokenizer


def performance_analysis():
    """

    The function is used to compare the action reorganization by the pre-trained model and fine-tuned model

    """
    model, tokenizer = load_model_and_tokenizer('bert-base-uncased', 3)
    model_v2, tokenizer_v2 = load_model_and_tokenizer('thameemk/actions-recognizer', 3)

    def compare_results(input_text):
        res = recognize_action(model, tokenizer, input_text)
        res_v2 = recognize_action(model_v2, tokenizer_v2, input_text)

        return {
            'input_text': input_text,
            'pre_trained': res.action.name,
            'fine_tuned': res_v2.action.name
        }

    return [compare_results("send the documentation to the team"), compare_results("download the game from store"),
            compare_results("seach for healthy food"), compare_results("download open source github projects")]


if __name__ == '__main__':
    resp = performance_analysis()
    print(resp)
