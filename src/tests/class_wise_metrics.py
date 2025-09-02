from sklearn.metrics import precision_score, recall_score, f1_score, balanced_accuracy_score
import numpy as np
from config_visual import *

def calculate_class_1_metrics(y_true, y_pred):
    """Calculate class 1 specific metrics for crisis prediction."""
    if len(y_true.shape) > 1:
        y_true_labels = np.argmax(y_true, axis=1)
        y_pred_labels = np.argmax(y_pred, axis=1)
    else:
        y_true_labels, y_pred_labels = y_true, y_pred
    
    metrics = {
        'class_1_f1': f1_score(y_true_labels, y_pred_labels, pos_label=1, zero_division=0),
        'class_1_precision': precision_score(y_true_labels, y_pred_labels, pos_label=1, zero_division=0),
        'class_1_recall': recall_score(y_true_labels, y_pred_labels, pos_label=1, zero_division=0),
        'class_1_support': np.sum(y_true_labels == 1),
        'class_0_f1': f1_score(y_true_labels, y_pred_labels, pos_label=0, zero_division=0),
        'class_0_support': np.sum(y_true_labels == 0),
        'overall_accuracy': np.mean(y_true_labels == y_pred_labels),
        'balanced_accuracy': balanced_accuracy_score(y_true_labels, y_pred_labels)
    }
    return metrics

def evaluate_crisis_prediction(georf, X_test, y_test, X_group_test, test_period="Test"):
    """Evaluate model with focus on crisis prediction metrics."""
    y_pred_test = georf.predict(X_test, X_group_test)
    
    # Calculate class-specific metrics
    class_1_metrics = calculate_class_1_metrics(y_test, y_pred_test)
    
    print("=" * 60)
    print(f"CRISIS PREDICTION EVALUATION - {test_period} (Class 1 Focus)")
    print("=" * 60)
    print(f"Class 1 F1 Score:      {class_1_metrics['class_1_f1']:.4f} [PRIMARY METRIC]")
    print(f"Class 1 Precision:     {class_1_metrics['class_1_precision']:.4f}")
    print(f"Class 1 Recall:        {class_1_metrics['class_1_recall']:.4f}")
    print(f"Class 1 Support:       {class_1_metrics['class_1_support']}")
    print("-" * 40)
    print(f"Class 0 F1 Score:      {class_1_metrics['class_0_f1']:.4f}")
    print(f"Class 0 Support:       {class_1_metrics['class_0_support']}")
    print("-" * 40)
    print(f"Overall Accuracy:      {class_1_metrics['overall_accuracy']:.4f}")
    print(f"Balanced Accuracy:     {class_1_metrics['balanced_accuracy']:.4f}")
    print("=" * 60)
    
    # Highlight the key insight
    if VIS_DEBUG_CRISIS_FOCUS:
        print("\nCRISIS PREDICTION FOCUS:")
        print(f"   - Optimizing for CLASS 1 F1: {class_1_metrics['class_1_f1']:.4f}")
        print(f"   - Crisis prediction accuracy: {class_1_metrics['class_1_precision']:.1%}")
        print(f"   - Crisis detection rate: {class_1_metrics['class_1_recall']:.1%}")
        print(f"   - Total crisis events: {class_1_metrics['class_1_support']}")
    
    return class_1_metrics, y_pred_test