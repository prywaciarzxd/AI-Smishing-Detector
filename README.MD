# SMS Preprocessing and Smishing Detection

This repository contains scripts for preprocessing SMS data to remove duplicates and normalize text, followed by creating features for machine learning models to classify smishing messages.

## Usage

### Requirements

- Python 3.x
- Required Python packages can be installed via:
pip install sckit-learn, tensorflow, kaggle
- Linux shell

- Check the user base path using the command:
```bash
python -m site --user-base
```
- Add to path with command:
```bash
export PATH=$PATH:<specific_path>
```
- U can export this path to be used everytime when running system by adding above line to file ~/.bashrc
- Then make dir in this location: ~/.kaggle move your json API file there and give it right permissions with command:
```bash
chmod 600 ~/.kaggle/kaggle.json
```
### Running the program

To execute the program, run the following command in your terminal:
```bash
python main.py

```

### Repository structure


- classifiers.py         # Script defining different classifiers for model training
- main.py                # Main script to execute preprocessing and classification
- message_label.py       # Script for labeling messages
- remove_duplicates.py   # Script for removing duplicate messages
- tokenik.py             # Script for creating features from tokenized messages
- SMSSmishCollection.txt # Download dataset from https://www.kaggle.com/datasets/galactus007/sms-smishing-collection-data-set

### Description
 - classifiers.py: Contains definitions of various classifiers used for training the machine learning models.
 - main.py: Entry point script that orchestrates preprocessing of SMS data, trains classifiers, and evaluates their performance.
 - message_label.py: Script for labeling messages as smishing or not.
 - remove_duplicates.py: Script to preprocess data by removing duplicate SMS messages.
 - tokenik.py: Script for tokenization and feature extraction from SMS messages.

### Workflow
 - Data Preprocessing: The SMS dataset undergoes preprocessing to remove duplicate messages and normalize text.
 - Feature Extraction: Tokenization is performed on preprocessed messages to extract features for machine learning models.
 - Model Training: Several classifiers (Naive Bayes, SVM, K-Nearest Neighbors, Decision Tree, Random Forest, Logistic Regression) are trained on the extracted features.
 - Evaluation: Each trained model is evaluated for accuracy, precision, recall, F1-score, False Positive Rate (FPR), and True Positive Rate (TPR).
 - Optimization: Models are sorted based on performance metrics to identify the top classifiers.
