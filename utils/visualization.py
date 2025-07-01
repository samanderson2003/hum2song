"""
Visualization utilities for training and performance analysis
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve, auc

def plot_training_curves(train_losses, train_accuracies, save_path=None):
    """Plot training loss and accuracy curves"""
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies)
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training curves saved to {save_path}")
    
    plt.show()

def plot_performance_analysis(results, save_path=None):
    """Create comprehensive performance visualization"""
    plt.figure(figsize=(15, 10))
    
    # Extract data for plotting
    all_preds = results['all_predictions']
    confidences = [p['confidence'] for p in all_preds]
    actual_labels = [1 if p['actual_match'] else 0 for p in all_preds]
    predicted_labels = [1 if p['predicted_match'] else 0 for p in all_preds]
    
    # 1. Confidence distribution
    plt.subplot(2, 3, 1)
    match_confidences = [p['confidence'] for p in all_preds if p['actual_match']]
    no_match_confidences = [p['confidence'] for p in all_preds if not p['actual_match']]
    
    plt.hist(match_confidences, alpha=0.7, label='Actual Matches', bins=20, color='green')
    plt.hist(no_match_confidences, alpha=0.7, label='Non-Matches', bins=20, color='red')
    plt.axvline(x=0.5, color='black', linestyle='--', label='Threshold (0.5)')
    plt.xlabel('Confidence Score')
    plt.ylabel('Frequency')
    plt.title('Confidence Score Distribution')
    plt.legend()
    
    # 2. ROC Curve
    plt.subplot(2, 3, 2)
    fpr, tpr, _ = roc_curve(actual_labels, confidences)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    
    # 3. Precision-Recall Curve
    plt.subplot(2, 3, 3)
    precision, recall, _ = precision_recall_curve(actual_labels, confidences)
    pr_auc = auc(recall, precision)
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AUC = {pr_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    
    # 4. Confusion Matrix
    plt.subplot(2, 3, 4)
    tp = len(results['true_positives'])
    fp = len(results['false_positives'])
    tn = len(results['true_negatives'])
    fn = len(results['false_negatives'])
    
    conf_matrix = np.array([[tn, fp], [fn, tp]])
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
               xticklabels=['Predicted No Match', 'Predicted Match'],
               yticklabels=['Actual No Match', 'Actual Match'])
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    
    # 5. Individual test results
    plt.subplot(2, 3, 5)
    song_names = list(set([p['song'] for p in all_preds]))
    hum_names = list(set([p['hum'] for p in all_preds]))
    
    # Create a matrix of confidence scores
    conf_matrix_detailed = np.zeros((len(hum_names), len(song_names)))
    for i, hum in enumerate(hum_names):
        for j, song in enumerate(song_names):
            # Find the prediction for this hum-song pair
            pred = next((p for p in all_preds if p['hum'] == hum and p['song'] == song), None)
            if pred:
                conf_matrix_detailed[i, j] = pred['confidence']
    
    sns.heatmap(conf_matrix_detailed, annot=True, fmt='.3f', cmap='RdYlGn',
               xticklabels=song_names, yticklabels=hum_names, vmin=0, vmax=1)
    plt.title('Confidence Scores: Hum vs Song')
    plt.xlabel('Songs')
    plt.ylabel('Hums')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    
    # 6. Performance summary
    plt.subplot(2, 3, 6)
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    values = [
        (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0,
        tp / (tp + fp) if (tp + fp) > 0 else 0,
        tp / (tp + fn) if (tp + fn) > 0 else 0,
        2 * (tp / (tp + fp) if (tp + fp) > 0 else 0) * (tp / (tp + fn) if (tp + fn) > 0 else 0) / 
        ((tp / (tp + fp) if (tp + fp) > 0 else 0) + (tp / (tp + fn) if (tp + fn) > 0 else 0)) 
        if ((tp / (tp + fp) if (tp + fp) > 0 else 0) + (tp / (tp + fn) if (tp + fn) > 0 else 0)) > 0 else 0
    ]
    
    bars = plt.bar(metrics, values, color=['blue', 'green', 'orange', 'red'])
    plt.ylim(0, 1)
    plt.title('Performance Metrics')
    plt.ylabel('Score')
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Performance analysis saved to {save_path}")
    
    plt.show()

def plot_confidence_histogram(results, save_path=None):
    """Plot histogram of confidence scores"""
    all_preds = results['all_predictions']
    match_confidences = [p['confidence'] for p in all_preds if p['actual_match']]
    no_match_confidences = [p['confidence'] for p in all_preds if not p['actual_match']]
    
    plt.figure(figsize=(10, 6))
    plt.hist(match_confidences, alpha=0.7, label='Actual Matches', bins=30, color='green', density=True)
    plt.hist(no_match_confidences, alpha=0.7, label='Non-Matches', bins=30, color='red', density=True)
    plt.axvline(x=0.5, color='black', linestyle='--', label='Threshold (0.5)')
    plt.xlabel('Confidence Score')
    plt.ylabel('Density')
    plt.title('Distribution of Confidence Scores')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confidence histogram saved to {save_path}")
    
    plt.show()
