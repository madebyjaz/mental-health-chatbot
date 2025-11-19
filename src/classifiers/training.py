# importing the necessary libraries
import argparse
from datasets import load_dataset as lds
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

# three inntende classification tasks: emotion, intent, risk
TASKS = {
    "emotion": {
        "model_name": "google/electra-base-goemotions",     # previously SamLowe/roberta-base-go_emotions could be google/electra-base-goemotions
        "num_labels": 28,                               # 27 emotion labels + 1 neutral
        "problem_type": "multilabel_classification",
        "dataset": "go_emotions",
        "metric": "f1-micro",
        "trainable": True,
    },

        "risk": {
        "model_name": "microsoft/deberta-v3-large",
        "num_labels": 2,                               # crisis/ self-harm vs non-crisis  
        "problem_type": "binary_classification",
        "dataset": "asmaab/dreaddit",
        "metric": "auroc",
        "trainable": True,
    },

    #intent classifier is zero-shot multi- natural lang. interface 
    # Therefore no training is needed for this particular classifier
}

def compute_metrics(eval_pred):

    if TASKS["dataset"] == "go_emotions":
        dataset = lds("go_emotions")