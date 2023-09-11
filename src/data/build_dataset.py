# Build a custom dataset for fine-tuning


def generate_custom_dataset() -> list[dict]:
    """

    Returns: List[Dict] -  Examples: [{'text': 'write for browse play', 'label': 'write'}, {'text': 'call for write
    browse learn', 'label': 'call'}]
    """
    import random

    # List of example verbs
    verbs = ["search", "send", "read", "write", "calculate", "browse", "translate", "call", "play", "learn"]

    # Generate 1000 random input and output pairs
    input_texts = []
    output_words = []

    for _ in range(1000):
        input_text = random.choice(verbs) + " for " + " ".join(random.sample(verbs, random.randint(1, 3)))
        output_word = input_text.split()[0]

        input_texts.append(input_text)
        output_words.append(output_word)

    # Print the generated pairs

    dataset = [{'text': input_texts[i], 'label': output_words[i]} for i in range(len(input_texts))]
    return dataset
