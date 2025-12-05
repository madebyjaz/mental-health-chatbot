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
        # Add validation split if not present
        if "validation" not in dataset:
            dataset["validation"] = dataset["test"]
    else:
        raise ValueError("‼️ The dataset your're trying to train on is not supported.")
    
    def preprocess(batch):
        tokenize = tokenizer(batch["text"], padding="max_length", truncation=True, max_length=256)
        if "labels" in batch:
            # convertin integers form lst to multi-hot float tensor for multi-label classification
            mh = []
            for label_list in batch["labels"]:
                # vec = [0.0] * 28                   Not the better version as it assumes the max 28 labels for go_emotions
                vec = [0.0] * cfg["num_labels"]
                for lbl in label_list:
                    vec[lbl] = 1.0
                mh.append(vec)
            tokenize["labels"] = mh

        return tokenize
    
    encoded_ds = dataset.map(preprocess, batched=True)

    cols = ["input_ids", "attention_mask"]
    if "labels" in encoded_ds["train"].column_names:
        cols.append("labels")

    encoded_ds.set_format(
        type="torch",
        columns=cols
    )

    targs = TrainingArguments(
        output_dir = output_dir + f"{task}/",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,                   # typical learning rate for fine-tuning transformers
        per_device_train_batch_size=8,         # should be 8 or 16 based on GPU memory
        per_device_eval_batch_size=8,
        num_train_epochs=5,                     # 5 for the intended classifiers; can be adjusted based on validation performance
        weight_decay=0.01,                      # to prevent overfitting
        load_best_model_at_end=True,            # load the best model when finished training (based on eval loss)
        metric_for_best_model=cfg["metric"],
        report_to="none",                       # disable default logging
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=encoded_ds["train"],
        eval_dataset=encoded_ds["validation"],
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(f"{output_dir}/{task}")
    print(f"✅ The Training for {task} classifier completed and model saved to {output_dir}/{task}.")
    tokenizer.save_pretrained(f"{output_dir}/{task}")
    print(f"✅ The Tokenizer for {task} classifier saved to {output_dir}/{task}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train classifiers for the mental health chatbot.")
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=TASKS.keys(),
        help="The classification task to train: emotion, risk.",
    )
    parser.add_argument(
        "--train_file",
        type=str,
        default=None,
        help="The path to a custom training data file (if any).",
    )
    parser.add_argument(
        "--val_file",
        type=str,
        default=None,
        help="The path to a custom validation data file (if any).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./models/",
        help="The directory to save the trained model and tokenizer.",
    )

    args = parser.parse_args()
    train(args.task, args.train_file, args.val_file, args.output_dir)