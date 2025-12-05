import argparse
import csv
import os
from typing import List, Dict, Tuple

from sklearn.metrics import (
    classification_report,
    precision_recall_fscore_support,
    accuracy_score,
    roc_auc_score,
)
from sklearn.preprocessing import LabelBinarizer

# Local classifiers
from src.classifiers.emotion_classifier import EmotionModel
from src.classifiers.intent_classifier import IntentModel
from src.classifiers.risk_classifier import RiskModel


def load_dataset(path: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"text", "emotion", "intent", "risk"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV must include columns: {', '.join(sorted(required))}. Missing: {', '.join(sorted(missing))}")
        for r in reader:
            rows.append({k: (r.get(k) or "").strip() for k in required})
    return rows


def evaluate_labels(y_true: List[str], y_pred: List[str], task_name: str) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    # Accuracy
    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    # Precision, Recall, F1 (micro)
    p_micro, r_micro, f_micro, _ = precision_recall_fscore_support(y_true, y_pred, average="micro", zero_division=0)
    metrics.update({"precision_micro": p_micro, "recall_micro": r_micro, "f1_micro": f_micro})
    # Macro (often useful for class-imbalanced sets)
    p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    metrics.update({"precision_macro": p_macro, "recall_macro": r_macro, "f1_macro": f_macro})

    # AUROC (one-vs-rest): only if we have class probability scores
    metrics["roc_auc_ovr_macro"] = float("nan")
    metrics["roc_auc_ovr_weighted"] = float("nan")

    return metrics


def maybe_compute_auroc(y_true: List[str], class_scores: List[Dict[str, float]]) -> Tuple[float, float]:
    # Convert to score matrix and binarized labels
    classes = sorted({c for scores in class_scores for c in scores.keys()})
    lb = LabelBinarizer()
    lb.fit(classes)
    y_true_bin = lb.transform(y_true)
    # Align scores to class order
    import numpy as np
    score_mat = np.array([[scores.get(c, 0.0) for c in lb.classes_] for scores in class_scores])
    try:
        auc_macro = roc_auc_score(y_true_bin, score_mat, average="macro", multi_class="ovr")
        auc_weighted = roc_auc_score(y_true_bin, score_mat, average="weighted", multi_class="ovr")
        return auc_macro, auc_weighted
    except Exception:
        return float("nan"), float("nan")


def evaluate_task(rows: List[Dict[str, str]], which: str) -> Dict[str, float]:
    if which == "emotion":
        model = EmotionModel()
        label_key = "emotion"
    elif which == "intent":
        model = IntentModel()
        label_key = "intent"
    elif which == "risk":
        model = RiskModel()
        label_key = "risk"
    else:
        raise ValueError(f"Unknown task: {which}")

    y_true: List[str] = []
    y_pred: List[str] = []
    # Optional: collect class scores if model provides them
    class_scores: List[Dict[str, float]] = []

    for r in rows:
        text = r["text"]
        y_true.append(r[label_key])
        # Predict label
        pred = model.predict(text)
        y_pred.append(pred)
        # If model has predict_proba or score, store per-class scores for AUROC
        scores: Dict[str, float] = {}
        if hasattr(model, "predict_proba"):
            try:
                s = model.predict_proba(text)
                if isinstance(s, dict):
                    scores = {str(k): float(v) for k, v in s.items()}
            except Exception:
                scores = {}
        class_scores.append(scores)

    metrics = evaluate_labels(y_true, y_pred, which)
    # Compute AUROC if we have usable scores
    if any(class_scores):
        auc_macro, auc_weighted = maybe_compute_auroc(y_true, class_scores)
        metrics["roc_auc_ovr_macro"] = auc_macro
        metrics["roc_auc_ovr_weighted"] = auc_weighted

    # Print a brief report
    print(f"\n=== {which.upper()} ===")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision (micro): {metrics['precision_micro']:.4f} | Recall (micro): {metrics['recall_micro']:.4f} | F1 (micro): {metrics['f1_micro']:.4f}")
    print(f"Precision (macro): {metrics['precision_macro']:.4f} | Recall (macro): {metrics['recall_macro']:.4f} | F1 (macro): {metrics['f1_macro']:.4f}")
    print(f"ROC-AUC OvR (macro): {metrics['roc_auc_ovr_macro']}")
    print(f"ROC-AUC OvR (weighted): {metrics['roc_auc_ovr_weighted']}")

    # Optional detailed per-class report
    print("\nPer-class report:")
    print(classification_report(y_true, y_pred, zero_division=0))
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate classifiers on a labeled CSV dataset.")
    parser.add_argument("--data", help="Path to CSV with columns: text, emotion, intent, risk. If omitted, uses bundled sample_eval.csv")
    parser.add_argument("--tasks", nargs="*", default=["emotion", "risk"], help="Subset of tasks to evaluate (default: emotion risk)")
    args = parser.parse_args()

    if args.data:
        data_path = args.data
    else:
        # Use bundled sample dataset next to this file
        here = os.path.dirname(__file__)
        data_path = os.path.join(here, "sample_eval.csv")
        print(f"No --data provided. Using bundled sample: {data_path}")

    rows = load_dataset(data_path)
    print(f"Loaded {len(rows)} examples from {data_path}")
    for which in args.tasks:
        evaluate_task(rows, which)


if __name__ == "__main__":
    main()
