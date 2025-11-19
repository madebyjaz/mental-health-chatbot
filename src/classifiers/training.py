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
        "model_name": "google/electra-base-discriminator",     # wrote the wrong model name for this classifier in the previous version
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
    predictions, labels = eval_pred
    preds = predictions.argmax(-1)

    ac = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average = 'weighted')
    metrics = {
        "accuracy": ac,
        "precision": precision, 
        "recall": recall, 
        "f1": f1,
    }

    try: 
        auroc = roc_auc_score(labels, predictions[:,1])
        metrics["auroc"] = auroc

    except ValueError:  
        raise ValueError("AUROC is not able to be computed for multiclass classification tasks.")
        pass

    return metrics

def train(task, train_file = None, val_file = None,  output_dir = "./models/"):

    cfg = TASKS[task]
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["model_name"],
        num_labels = cfg["num_labels"],
    )

    if cfg["dataset"] == "go_emotions":
        dataset = lds("go_emotions")
    elif cfg["dataset"] == "asmaab/dreaddit":
        dataset = lds("asmaab/dreaddit")
    else:
        raise ValueError("‼️ The dataset your're trying to train on is not supported.")
    
    def preorocess(batch):
        tokenize = tokenizer(batch["text"], padding="max_length", truncation=True, max_length=256)
        if "labels" in batch:
            tokenize["labels"] = batch["labels"]
        return tokenize
    
    encoded_ds = dataset.map(preorocess, batched=True)

    cols = ["input_ids", "attention_mask"]
    if "labels" in encoded_ds["train"].column_names:
        cols.append("labels")

    encoded_ds.set_format(
        type="torch",
        columns=cols
    )


