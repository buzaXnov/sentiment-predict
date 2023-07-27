import re
import unicodedata

import torch
import emoji
from transformers import AutoTokenizer, AutoModelForSequenceClassification

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class SentimentClassifier(torch.nn.Module):
    def __init__(self, tokenizer, model, max_length=512):
        """Initialize the model."""
        super().__init__()

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.model = model

        # Optional: If you want to use a specific device (e.g., 'cuda' or 'cpu')
        self.model.to(device)

        self.label_map = {
            0: 'Extremely negative',
            1: 'Slightly negative',
            2: 'Neutral',
            3: 'Slightly positive',
            4: 'Extremely positive'
        }

    def forward(self, inputs):
        """Forward pass of the model."""
        # Get the output of the model
        outputs = self.model(**inputs)

        return outputs

    def predict(self, input_text):

        input_text = self.preprocess(input_text)

        # Tokenize the input text
        inputs = self.tokenizer.encode_plus(
            input_text,
            None,
            add_special_tokens=True,
            padding='max_length',
            max_length=128,
            return_token_type_ids=False,
            return_attention_mask=True,
            return_tensors='pt'
        )

        # Move the inputs to the specified device
        inputs = {key: val.to(device) for key, val in inputs.items()}

        # Perform inference
        with torch.no_grad():
            outputs = self(inputs)

        # Get the predicted logits and probabilities
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=-1)

        # Get the predicted class (index of the class with the highest probability)
        predicted_class = torch.argmax(probabilities, dim=-1).item()

        return predicted_class, self.label_map[predicted_class]

    @staticmethod
    def preprocess(text, remove_dots=True):
        """Preprocess the input text to be a bare lowercased text."""
        # Remove emojis
        text = emoji.demojize(text)

        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)

        # Remove hashtags
        text = re.sub(r'#\w+', '', text)

        # Remove tags (e.g., @username)
        text = re.sub(r'@\w+', '', text)

        # Remove newlines
        text = re.sub(r'\n', ' ', text)

        # Remove lines with dashes
        text = re.sub(r'[-]{2,}', '', text)

        # Remove all punctuations except the full stop (.)
        text = re.sub(r'[^\w\s.]', '', text)

        if remove_dots:
            text = re.sub(r'\.', '', text)

        # Convert to normal unicode text
        text = unicodedata.normalize('NFKD', text).encode(
            'ascii', 'ignore').decode('utf-8')

        # Convert to lowercase (optional, depending on your use case)
        text = text.lower()

        # Remove whitespaces
        text = text.strip()

        return text


def main(input_text):

    # Initialize the model
    model_name_or_path = './trained_model'

    # Load the tokenizer and model
    # tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-mini")
    tokenizer = AutoTokenizer.from_pretrained(
        f"{model_name_or_path}/tokenizer", local_files_only=True)
    model_params = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path, local_files_only=True)

    model = SentimentClassifier(tokenizer, model_params)
    predicted_class, predicted_label = model.predict(input_text)

    # Example output
    print("Input Text:", input_text)
    print("Predicted Class:", predicted_class)
    print("Predicted Label:", predicted_label)


if __name__ == "__main__":
    text = "\n\nThis is an example sentence\n for inference..... (()()(/()))"
    main(text)

# TODO: Funkcija u py file koji vraca object koji sadrzi i int i string labele.
# U istom folderu staviti model i py skriptu; preprocessing file takoder za input stringove

# NOTE: Situacija gdje je manje od x riječi ili gdje je post samo emoji???
# NOTE: Should I remove numbers?
# NOTE: Kako će model primati inpute? Hoće li rezultate vraćati kao return ili će ih samo printati?
