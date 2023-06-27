# Arabic Natural Language Inference (NLI) Project

## Introduction
This project is focused on performing Natural Language Inference (NLI) tasks on Arabic datasets (ArNLi and ArbTEDS). The objective of our work is to present a comprehensive study and benchmark comparison of different pre-trained models in these NLP tasks for Arabic language. Furthermore, we propose and experiment with different loss functions to enhance model performance and robustness.

## installation 
Clone this repository and install the required Python dependencies.

```
git clone https://github.com/yourusername/arabic-nli.git
cd arabic-nli
pip install -r requirements.txt
```

## Data
The datasets used in this project, ArNLi and ArbTEDS, can be found in the `dataset/dataset/ArbTEDS/` directory.

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
 --sentence1 "البغدادي المحمودي يعرض على المعارضة وقفا ل إطلاق النار و يقول إن طرابلس مستعدة ل لحوار مع ها و ذلك في إشارة إلى أن أكثر من 100 يوم من القتال الدائر في البلاد و القصف الكثيف ل قوات الناتو قد يكونان أفلحا ب انتزاع تنازلات من العقيد القذافي" \
 --sentence2 "رئيس الحكومة الليبية مستعدون ل وقف إطلاق النار و الحوار مع المعارضة"
```

## License
This project is licensed under the terms of the MIT license.

## Citations
If you use this code or our results in your research, please cite:

```
@misc{yourname2023arabicnli,
  title={Enhancing Arabic Natural Language Inference using Pre-Trained Transformer Models: A Comparative Study and a New Benchmark},
  author={Fouzi Takelait},
  year={2023},
  eprint={12345},
  archivePrefix={arXiv},
  primaryClass={cs.CL} 
}
```
