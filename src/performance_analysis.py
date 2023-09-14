from transformers import AutoTokenizer, AutoModelForSequenceClassification, BertForSequenceClassification, Trainer
import torch


def performance_analysis():
    tokenizer_v2 = AutoTokenizer.from_pretrained('search_action')
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    text = "search for some healthy food"
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors='pt')
    inputs_v2 = tokenizer_v2(text, padding=True, truncation=True, return_tensors='pt')

    model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=3)
    model_v2 = BertForSequenceClassification.from_pretrained("search_action", num_labels=3)

    predictions = torch.nn.functional.softmax(model(**inputs).logits, dim=-1).cpu().detach().numpy()

    predictions_v2 = torch.nn.functional.softmax(model_v2(**inputs_v2).logits, dim=-1).cpu().detach().numpy()

    print(predictions, predictions_v2)


if __name__ == '__main__':
    performance_analysis()
