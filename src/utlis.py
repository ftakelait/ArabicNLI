from datasets import load_dataset, load_metric, Dataset
from loss_function import FocalLoss
import pandas as pd
import numpy as np
from transformers import Trainer, TrainingArguments


def upload_dataset(train_file: str, valid_file: str, test_file: str) -> Dataset:
    """
    This function is used to upload a dataset for a given CSV file.

    Parameters:
    train_file (str): The path of the training data file in csv format.
    valid_file (str): The path of the validation data file in csv format.
    test_file (str): The path of the test data file in csv format.

    Returns:
    dataset (Dataset): A `datasets.Dataset` object with train, validation, and test splits.
    """

    # Load the training dataset
    dataset = load_dataset("csv", data_files=train_file)

    # Load the validation and test datasets as pandas DataFrame and convert them to `datasets.Dataset`
    val_data = pd.read_csv(valid_file)
    ds_val = Dataset.from_pandas(val_data)

    test_data = pd.read_csv(test_file)
    ds_test = Dataset.from_pandas(test_data)

    # Add the validation and test datasets to the main dataset
    dataset["validation"] = ds_val
    dataset["test"] = ds_test

    return dataset


# Initialize metrics
acc_metric = load_metric('accuracy')
f1_metric = load_metric('f1')
precision_metric = load_metric('precision')
recall_metric = load_metric('recall')


def compute_metrics(eval_pred: tuple) -> dict:
    """
    This function computes the accuracy, F1 score, precision, and recall of the model's predictions.

    Parameters:
    eval_pred (tuple): A tuple of (predictions, labels)

    Returns:
    metrics (dict): A dictionary containing accuracy, F1 score, precision, and recall.
    """

    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    accuracy = acc_metric.compute(predictions=predictions, references=labels)
    f1 = f1_metric.compute(predictions=predictions, references=labels,average='macro')
    precision = precision_metric.compute(predictions=predictions, references=labels,average='macro')
    recall = recall_metric.compute(predictions=predictions, references=labels,average='macro')

    return {"accuracy":accuracy['accuracy'],"f1":f1['f1'],"precision":precision['precision'],"recall":recall['recall']}


class CustomTrainer(Trainer):
    """
    Custom trainer for handling custom loss function. Inherits from `transformers.Trainer`.
    """
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Overrides `compute_loss` method from Trainer to include the Focal loss.
        """

        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        # Custom Focal loss
        loss_fct = FocalLoss(alpha=0.5, gamma=2.0, reduction='mean')
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))

        return (loss, outputs) if return_outputs else loss
