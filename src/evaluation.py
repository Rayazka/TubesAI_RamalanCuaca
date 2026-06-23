def calculate_confusion_matrix(y_true, y_pred):
    """
    Calculates True Positives, True Negatives, False Positives, and False Negatives.
    
    Args:
        y_true (list): Actual target labels.
        y_pred (list): Predicted labels.
        
    Returns:
        tuple: (tp, tn, fp, fn) counts.
    """
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    
    for true, pred in zip(y_true, y_pred):
        if true == 1 and pred == 1:
            tp += 1
        elif true == 0 and pred == 0:
            tn += 1
        elif true == 0 and pred == 1:
            fp += 1
        elif true == 1 and pred == 0:
            fn += 1
            
    return tp, tn, fp, fn

def calculate_metrics(y_true, y_pred):
    """
    Computes Accuracy, Precision, Recall, and F1-Score from scratch.
    
    Args:
        y_true (list): Actual target labels.
        y_pred (list): Predicted labels.
        
    Returns:
        dict: Dictionary containing the metrics.
    """
    total = len(y_true)
    if total == 0:
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "confusion_matrix": (0, 0, 0, 0)
        }
        
    tp, tn, fp, fn = calculate_confusion_matrix(y_true, y_pred)
    
    accuracy = (tp + tn) / total
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "confusion_matrix": (tp, tn, fp, fn)
    }

def print_metrics(metrics, title="Evaluation Results"):
    """
    Nicely formats and prints evaluation metrics and the confusion matrix.
    
    Args:
        metrics (dict): Computed metrics from calculate_metrics.
        title (str): Header title.
    """
    tp, tn, fp, fn = metrics["confusion_matrix"]
    
    print(f"\n==============================================")
    print(f"  {title.upper()}")
    print(f"==============================================")
    print(f" Accuracy  : {metrics['accuracy']:.4f} ({metrics['accuracy'] * 100:.2f}%)")
    print(f" Precision : {metrics['precision']:.4f} ({metrics['precision'] * 100:.2f}%)")
    print(f" Recall    : {metrics['recall']:.4f} ({metrics['recall'] * 100:.2f}%)")
    print(f" F1-Score  : {metrics['f1_score']:.4f} ({metrics['f1_score'] * 100:.2f}%)")
    print(f"----------------------------------------------")
    print(f" Confusion Matrix:")
    print(f"                    Predicted Hujan    Predicted Tidak Hujan")
    print(f" Actual Hujan            {tp:<15}    {fn:<15}")
    print(f" Actual Tidak Hujan      {fp:<15}    {tn:<15}")
    print(f"==============================================")
