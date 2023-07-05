# Arabic Natural Language Inference (NLI) System Using Transformer Models

## Introduction
Natural Language Inference (NLI), also known as Recognizing Textual Entailment (RTE), plays a pivotal role in the field of Natural Language Processing (NLP). The task involves determining the logical relationship between two text pieces, typically referred to as the premise and hypothesis. NLI has profound implications in various NLP and AI applications, including dialogue improvement, question answering, machine translation, text classification, summarization, and information retrieval.

Despite its importance, limited research has been done on Arabic NLI, particularly using Neural Networks (NN). Our project aims to address this research gap by proposing a novel approach that leverages pre-trained transformer models for the NLI task on Arabic datasets.

## Project Overview
Our approach employs pre-trained transformer models, renowned for their ability to capture both lexical and semantic features, thus offering the potential to surpass traditional NLI methods. By treating the inference task as a classification problem, our system eliminates the need for manual feature engineering, thus simplifying the process while maximizing efficiency.

Key contributions of our project include:

- Introduction of an NLI system for Arabic, leveraging pre-trained transformer models.
- Evaluation of different types of pre-trained transformer models and loss functions to enhance the system's robustness.
- Comprehensive performance evaluation of our proposed system using different Arabic NLI datasets.

Our results are compared against existing Arabic NLI benchmarks to measure the effectiveness of our approach.

This repository contains all the relevant codes, datasets, and resources used in our project. It also includes a detailed description of the methodology, experimental results, and potential future research directions.

This project contributes significantly to the ongoing research in Arabic NLI, providing a practical, efficient, and potentially more effective solution for the task. We welcome researchers and developers to use our work as a basis for further research and development in the field.

## installation 
Clone this repository and install the required Python dependencies.

```
git clone https://github.com/ftakelait/ArabicNLI.git
cd ArabicNLI
pip install -r requirements.txt
```

## Data
This project utilizes two significant Arabic NLI datasets, namely ArNLi and ArbTEDS.

- ArNLi
is an extensive dataset that has been used as a benchmark in the field of Arabic Natural Language Inference.

- ArbTEDS
is another widely accepted dataset, which has been specifically curated for Arabic NLI tasks.

Both of these datasets are fundamental to our project and have been instrumental in training and evaluating our models.

You can find these datasets in the `dataset/` directory of this repository. Please refer to their respective documentation for more information about their structure and content.

## Training

Run the following command to reproduce our results:

```
python train.py --model_name 'UBC-NLP/MARBERT' \
 --train_file 'dataset/dataset/ArbTEDS/train_ArbTEDS.csv' \
 --validation_file 'dataset/dataset/ArbTEDS/validation_ArbTEDS.csv' \
 --test_file 'dataset/dataset/ArbTEDS/test_ArbTEDS.csv' \
 --epochs 1 \
 --batch_size 16 \
 --output_dir 'checkpoints' \
 --log_file 'logs_file.log' \
 --max_length 90 \
 --num_labels 2 \
 --loss_function 'focal_loss'
```
This script trains a model using the MARBERT pre-trained model on the ArbTEDS dataset with the specified configuration.

## Inference
After training the model, you can use it for inference with the following command:

```
python inference.py --checkpoints 'checkpoints/checkpoint-27/' \
 --batch_size 16 \
 --sentence1 "في الوقت الذي تعاني فيه العديد من المدن العربية من تلوث الهواء وضعف جودة الحياة ، تعتبر دبي نموذجاً للتنمية الحضرية المستدامة والحياة الرفيعة." \
 --sentence2 "دبي مدينة نظيفة ومزدهرة تعد مثالاً في التنمية الحضرية المستدامة."
```

## License
This project is licensed under the terms of the MIT license.

## Citations
**Note**: I am considering submitting my work to AAAI 2023 conference, but still not sure what conference I may submit to. Any feedback or guidelines will be much appreciated. 

If you use this code or our results in your research, please cite:

```
@misc{yourname2023arabicnli,
  title={Arabic Natural Language Inference with Pre-trained Transformers: Exploring the Power of Semantic Understanding},
  author={Fouzi Takelait},
  year={2023},
  eprint={12345},
  archivePrefix={arXiv},
  primaryClass={cs.CL} 
}
```
