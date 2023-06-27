"""
This module trains a transformer model for NLP tasks.
Provides utilities for training including custom loss functions, metrics, etc.
@author: Fouzi Takelait
"""

import argparse
import os
import logging
from typing import Dict
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from utils import upload_dataset, compute_metrics
from loss_function import CustomTrainer

def parse_args() -> argparse.Namespace:
    """
    Parse input arguments for script.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_name', type=str, help='Name of the pre-trained model.')
    parser.add_argument('--train_file', type=str, help='Path to the train data file.')
    parser.add_argument('--validation_file', type=str, help='Path to the validation data file.')
    parser.add_argument('--test_file', type=str, help='Path to the test data file.')
    parser.add_argument('--epochs', type=int, default=1, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training.')
    parser.add_argument('--output_dir', type=str, help='Directory for saving model checkpoints.')
    parser.add_argument('--log_file', type=str, help='Path to the log file.')
    parser.add_argument('--loss_function', type=str, default='cross_entropy', help='Type of loss function to use.')
    parser.add_argument('--learning_rate', type=float, default=0.00002, help='Learning rate.')
    parser.add_argument('--max_length', type=int, default=200, help='Maximum input length for the model.')
    parser.add_argument('--num_labels', type=int, help='Number of output labels.')

    return parser.parse_args()

def preprocess_function(examples: Dict, tokenizer: AutoTokenizer, max_length: int) -> Dict:
    """
    Function to preprocess examples for the model.
    """
    return tokenizer(examples['t'], examples['h'], truncation=True, padding="max_length", max_length=max_length)

def main() -> None:
    # Parse script arguments
    args = parse_args()

    # Setup logging
    logging.basicConfig(filename=args.log_file, level=logging.DEBUG)

    print("\n"*2)

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=args.num_labels)

    # Load and preprocess dataset
    dataset = upload_dataset(args.train_file, args.validation_file, args.test_file)
    print("\n"*2)
    print("ENCODING Dataset")
    encoded_dataset = dataset.map(lambda examples: preprocess_function(examples, tokenizer, args.max_length))

    # Setup training arguments
    training_args = TrainingArguments(
        args.output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        load_best_model_at_end=True,
    )

    # Initialize trainer
    if args.loss_function == 'focal_loss':
        print("#######")
        print("You are using Focal Loss")
        trainer_class = CustomTrainer
    else:
        trainer_class = Trainer

    trainer = trainer_class(
        model,
        training_args,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    print("\n"*2)

    # Train model
    trainer.train()

    # Evaluate model
    predictions = trainer.predict(encoded_dataset["test"])
    print("\nClassification report On Test Set")
    print("Accuracy: ", accuracy_score(predictions.label_ids, predictions.predictions.argmax(-1)))
    print("F1: ", f1_score(predictions.label_ids, predictions.predictions.argmax(-1), average='macro'))
    print("Precision: ", precision_score(predictions.label_ids, predictions.predictions.argmax(-1), average='macro'))
    print("Recall: ", recall_score(predictions.label_ids, predictions.predictions.argmax(-1), average='macro'))
    print()

    logging.debug(f'Prediction On testset {predictions.predictions.argmax(-1)}')

if __name__ == "__main__":
    main()
