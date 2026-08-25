"""Classification metrics for Gender and Age.

Computes accuracy, precision, recall, F1, confusion matrix, unknown rate, and coverage.
"""

from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from evaluation.pipeline.runner import EvalRecord


def compute_gender_metrics(records: list[EvalRecord]) -> dict:
    """Compute classification metrics for gender."""
    y_true = []
    y_pred = []
    
    unknowns = 0
    total = 0
    
    for r in records:
        if r.gt_gender:
            total += 1
            if r.pred_gender == "unknown":
                unknowns += 1
            else:
                y_true.append(r.gt_gender)
                y_pred.append(r.pred_gender)
                
    if total == 0:
        return {}
        
    unknown_rate = unknowns / total
    coverage = 1.0 - unknown_rate
    
    if not y_true:
         return {
            "unknown_rate": unknown_rate,
            "coverage": coverage,
        }
        
    acc = accuracy_score(y_true, y_pred)
    
    # We use macro average for overall F1, but we can also get precision/recall
    # We specify labels=["male", "female"] to ensure consistent ordering
    labels = ["male", "female"]
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average=None, zero_division=0)
    macro_f1 = sum(f1) / len(labels)
    
    # Confusion matrix manually for simplicity and guaranteed 2x2 shape
    cm = {"male_as_male": 0, "male_as_female": 0, "female_as_female": 0, "female_as_male": 0}
    for t, p in zip(y_true, y_pred):
        if t == "male" and p == "male":
            cm["male_as_male"] += 1
        elif t == "male" and p == "female":
            cm["male_as_female"] += 1
        elif t == "female" and p == "female":
            cm["female_as_female"] += 1
        elif t == "female" and p == "male":
            cm["female_as_male"] += 1
            
    return {
        "accuracy": acc,
        "precision_male": precision[0],
        "precision_female": precision[1],
        "recall_male": recall[0],
        "recall_female": recall[1],
        "f1_male": f1[0],
        "f1_female": f1[1],
        "macro_f1": macro_f1,
        "confusion_matrix": cm,
        "unknown_rate": unknown_rate,
        "coverage": coverage,
        "total_eval_samples": total
    }


def compute_age_metrics(records: list[EvalRecord]) -> dict:
    """Compute classification metrics for age brackets."""
    y_true = []
    y_pred = []
    
    unknowns = 0
    total = 0
    
    labels = ["18-30", "31-45", "46-60", "60+"]
    
    for r in records:
        if r.gt_age:
            total += 1
            if r.pred_age == "unknown":
                unknowns += 1
            else:
                y_true.append(r.gt_age)
                y_pred.append(r.pred_age)
                
    if total == 0:
        return {}
        
    unknown_rate = unknowns / total
    coverage = 1.0 - unknown_rate
    
    if not y_true:
        return {
            "unknown_rate": unknown_rate,
            "coverage": coverage,
        }
        
    acc = accuracy_score(y_true, y_pred)
    _, _, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average=None, zero_division=0)
    macro_f1 = sum(f1) / len(labels)
    
    # Confusion matrix
    cm = {f"{t}_as_{p}": 0 for t in labels for p in labels}
    for t, p in zip(y_true, y_pred):
        if t in labels and p in labels:
             key = f"{t}_as_{p}"
             cm[key] = cm.get(key, 0) + 1
             
    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "confusion_matrix": cm,
        "unknown_rate": unknown_rate,
        "coverage": coverage,
        "total_eval_samples": total
    }
