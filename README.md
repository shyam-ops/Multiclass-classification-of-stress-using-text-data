# Stress Level Classification using BERT

A machine learning project that classifies text into one of three stress levels: **Low**, **Moderate**, or **High**.  
The model is built using **BERT (bert-base-uncased)** and trained on a labeled text dataset of social posts.

## Project Overview

This project performs text classification for stress detection from user-written sentences.  
It preprocesses the text, balances the dataset using oversampling, tokenizes the input with BERT tokenizer, and fine-tunes a BERT sequence classification model.

## Dataset

The dataset used in this project is `Stress.csv`.

### Key columns
- `text`: input sentence or post
- `label`: target class
- `subreddit`, `post_id`, `sentence_range`, `confidence`, `social_timestamp`: additional metadata

### Class labels
The model predicts three classes:
- Low
- Moderate
- High

## Workflow

1. Load the dataset
2. Convert text to lowercase
3. Balance the dataset using oversampling
4. Split data into training and testing sets
5. Tokenize text using `bert-base-uncased`
6. Fine-tune BERT for sequence classification
7. Evaluate the model using classification metrics
8. Run inference on new user input

## Model Details

- Base model: `bert-base-uncased`
- Task: Multi-class text classification
- Number of labels: 3
- Max token length: 128
- Train/test split: 80/20
- Training epochs: 20
- Batch size: 8
- Weight decay: 0.01

## Results

The trained model achieved the following test performance:

- Accuracy: 0.90
- Low: Precision 0.96, Recall 0.88, F1-score 0.92
- Moderate: Precision 0.85, Recall 0.91, F1-score 0.88
- High: Precision 0.89, Recall 0.90, F1-score 0.89

## Example Predictions

- "I feel relaxed and happy today." → Low
- "I have so many deadlines coming up, it's getting stressful." → High
- "I don't know if I can finish everything on time." → High

## Tech Stack

- Python
- Pandas
- Scikit-learn
- PyTorch
- Hugging Face Transformers
- BERT

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/stress-level-classification.git
cd stress-level-classification
```

Install the required libraries:

```bash
pip install pandas scikit-learn torch transformers
```

## Usage

Open the notebook and run the cells step by step:

```bash
jupyter notebook Final_Bert.ipynb
```

Or open it in Google Colab and run it directly.

To test a new sentence, use the inference function:

```python
classify_stress_level("I am overwhelmed with work and deadlines.")
```

## Project Structure

```text
.
├── Final_Bert.ipynb
├── Stress.csv
└── README.md
```

## Notes

- The notebook balances the dataset using oversampling before training.
- The model uses the BERT tokenizer with padding and truncation.
- If Weights & Biases prompts appear during training, they can be configured or disabled depending on your environment.

## Future Improvements

- Add a web app or API for live stress prediction
- Tune hyperparameters for better performance
- Use a larger and more diverse dataset
- Add visualizations for class distribution and evaluation results
