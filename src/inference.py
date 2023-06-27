"""
This module performs inference on given sentences using a pre-trained model checkpoint.
@author: Fouzi Takelait
"""

import argparse
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import Tuple, Union

def parse_args() -> argparse.Namespace:
    """
    Parse input arguments for script.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--checkpoints', type=str, help='Path to the pre-trained model checkpoint.')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for inference.')
    parser.add_argument('--max_length', type=int, default=200, help='Maximum input length for the model.')
    parser.add_argument('--sentence1', type=str, help='Text of the first sentence.')
    parser.add_argument('--sentence2', type=str, help='Text of the second sentence.')

    return parser.parse_args()

def inference(sentence1: str, sentence2: str, model: AutoModelForSequenceClassification, tokenizer: AutoTokenizer) -> str:
    """
    Perform inference on a pair of sentences.
    """
    # Tokenize the sentences
    encoded_text = tokenizer(sentence1, sentence2, truncation=True, padding="max_length", return_tensors='pt')

    # Compute model output
    output = model(**encoded_text)

    # Convert logits to probabilities
    probabilities = torch.softmax(output.logits, dim=1).squeeze()

    # Get the predicted label
    label = torch.argmax(probabilities).item()

    # Define a mapping of labels to their meanings
    label_mapping = {0: 'No Entails', 1: 'Entails', 2: 'Neutral'}

    # Return the meaning of the predicted label
    return label_mapping[label]

def main() -> None:
    # Parse script arguments
    args = parse_args()

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoints, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.checkpoints)

    # Perform inference and print result
    result = inference(args.sentence1, args.sentence2, model, tokenizer)
    print("\nPrediction is: ", result)

if __name__ == '__main__':
    main()
