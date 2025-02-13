import os
import pandas as pd
import numpy as np

from sklearn.metrics import roc_auc_score,classification_report, balanced_accuracy_score
from sklearn.metrics import  accuracy_score, f1_score, roc_auc_score, confusion_matrix



def generate_metrics(result_df):

    metrics = []
    class_report = classification_report(result_df["y_true"], result_df["y_pred"], output_dict=True)
    
    # Overall metrics
    metric = {
        "pid":result_df['pid'].unique()[0],
        "accuracy": class_report['accuracy'],
        "balanced_accuracy": balanced_accuracy_score(result_df['y_true'], result_df['y_pred']),
        "auc":roc_auc_score(result_df['y_true'], result_df['y_prod']),
        **{"macro_" + key: val for key, val in class_report['macro avg'].items()},
        **{"weighted_" + key: val for key, val in class_report['weighted avg'].items()}
    }

    label_metrics = {str(label): values for label, values in class_report.items() if label in ["0", "1"]}
    for label, metrics_dict in label_metrics.items():
        metric.update({
            f"{label}_precision": metrics_dict['precision'],
            f"{label}_recall": metrics_dict['recall'],
            f"{label}_f1-score": metrics_dict['f1-score'],
            f"{label}_support": metrics_dict['support']
        })
    metrics.append(metric)
    
    # Overall metric value
    metrics_df = pd.DataFrame(metrics)
    return metrics_df


def save_metric(metrics_df,file_path):
    with open(file_path, 'w') as file:
        metrics_df.to_csv(file, mode='a')


