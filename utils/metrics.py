# utils/metrics.py
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import precision_recall_curve, f1_score
import logging

def calculate_metrics(labels, scores):
    """Calculate various metrics for anomaly detection performance"""
    # Find optimal threshold using F1 score
    precisions, recalls, thresholds = precision_recall_curve(labels, scores)
    f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-8)
    optimal_threshold = thresholds[np.argmax(f1_scores)]
    
    # Calculate metrics
    predictions = (scores > optimal_threshold).astype(int)
    metrics = {
        'auroc': roc_auc_score(labels, scores),
        'avg_precision': average_precision_score(labels, scores),
        'f1': f1_score(labels, predictions),
        'threshold': optimal_threshold
    }
    
    return metrics
