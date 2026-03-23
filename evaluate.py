"""
=============================================================================
EVALUATE.PY - Model Evaluation and Performance Metrics
=============================================================================

This module evaluates the trained Deepfake Detection model on the test set
and computes comprehensive metrics to assess performance.

EVALUATION METRICS EXPLAINED:
-----------------------------

1. ACCURACY:
   - Formula: (TP + TN) / (TP + TN + FP + FN)
   - What it means: Percentage of ALL predictions that are correct
   - Limitation: Can be misleading with imbalanced datasets
   - Example: 90% accuracy with 90% real images could mean predicting all real

2. PRECISION:
   - Formula: TP / (TP + FP)
   - What it means: Of all images predicted as FAKE, how many are actually fake?
   - High precision = Few false alarms (real images wrongly labeled as fake)
   - Important when: False positives are costly (e.g., accusing someone of deepfake)

3. RECALL (Sensitivity):
   - Formula: TP / (TP + FN)
   - What it means: Of all ACTUAL fake images, how many did we catch?
   - High recall = Few missed fakes
   - Important when: Missing a deepfake is costly (e.g., security applications)

4. F1 SCORE:
   - Formula: 2 * (Precision * Recall) / (Precision + Recall)
   - What it means: Harmonic mean of precision and recall
   - Balanced measure that penalizes extreme precision/recall imbalance
   - Range: 0 to 1 (higher is better)

5. CONFUSION MATRIX:
   - 2x2 grid showing:
     - True Negatives (TN): Real correctly classified as Real
     - False Positives (FP): Real incorrectly classified as Fake
     - False Negatives (FN): Fake incorrectly classified as Real
     - True Positives (TP): Fake correctly classified as Fake
   
   Visual representation:
                     Predicted
                   Real    Fake
   Actual  Real  [  TN  |  FP  ]
          Fake  [  FN  |  TP  ]

WHERE:
- TP (True Positive): Correctly identified fakes
- TN (True Negative): Correctly identified real images
- FP (False Positive): Real images wrongly labeled as fake (Type I error)
- FN (False Negative): Fake images wrongly labeled as real (Type II error)
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from model import get_model
from train import DeepfakeDataset, get_transforms, set_seed


# =============================================================================
# EVALUATION FUNCTIONS
# =============================================================================

def load_model(model_path: str, device: torch.device) -> nn.Module:
    """
    Load a trained model from checkpoint.
    
    Args:
        model_path (str): Path to the saved model weights (.pth file)
        device (torch.device): Device to load the model to
    
    Returns:
        nn.Module: Loaded model ready for inference
    """
    print(f"📂 Loading model from: {model_path}")
    
    # Initialize model architecture
    model = get_model(pretrained=False, freeze_backbone=False)
    
    # Load saved weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    # Move to device and set to evaluation mode
    model = model.to(device)
    model.eval()
    
    print("✓ Model loaded successfully!")
    return model


def get_test_loader(
    dataset_path: str,
    batch_size: int = 32,
    test_split: float = 0.1,
    val_split: float = 0.1,
    seed: int = 42
) -> DataLoader:
    """
    Create test data loader using the same split as training.
    
    IMPORTANT: We use the same seed and split ratios as training to ensure
    we evaluate on the exact same test set that was held out during training.
    
    Args:
        dataset_path (str): Path to dataset directory
        batch_size (int): Batch size for evaluation
        test_split (float): Same test split used in training
        val_split (float): Same validation split used in training
        seed (int): Same seed used in training
    
    Returns:
        DataLoader: Test data loader
    """
    # Create dataset
    full_dataset = DeepfakeDataset(dataset_path, transform=get_transforms(is_training=False))
    labels = full_dataset.labels
    indices = list(range(len(full_dataset)))
    
    # Reproduce the same split as training
    train_val_indices, test_indices = train_test_split(
        indices,
        test_size=test_split,
        stratify=labels,
        random_state=seed
    )
    
    # Create test subset
    test_subset = Subset(full_dataset, test_indices)
    
    # Create test loader
    test_loader = DataLoader(
        test_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"✓ Test loader created with {len(test_indices)} samples")
    return test_loader


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate model on test set and collect predictions.
    
    Args:
        model: Trained model
        test_loader: Test data loader
        device: Device for computation
    
    Returns:
        tuple: (all_labels, all_predictions, all_probabilities)
            - all_labels: Ground truth labels (0 or 1)
            - all_predictions: Binary predictions (0 or 1)
            - all_probabilities: Raw probability scores [0, 1]
    """
    print("\n🔄 Evaluating model on test set...")
    
    model.eval()
    
    all_labels = []
    all_predictions = []
    all_probabilities = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images = images.to(device)
            
            # Get model outputs (probabilities)
            outputs = model(images).squeeze()
            
            # Convert to binary predictions
            predictions = (outputs > 0.5).float()
            
            # Store results
            all_labels.extend(labels.numpy())
            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(outputs.cpu().numpy())
    
    return (
        np.array(all_labels),
        np.array(all_predictions),
        np.array(all_probabilities)
    )


def compute_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray
) -> Dict[str, float]:
    """
    Compute comprehensive evaluation metrics.
    
    Args:
        labels: Ground truth labels
        predictions: Binary predictions
        probabilities: Raw probability scores
    
    Returns:
        dict: Dictionary containing all metrics
    """
    metrics = {}
    
    # Basic metrics
    metrics['accuracy'] = accuracy_score(labels, predictions)
    metrics['precision'] = precision_score(labels, predictions, zero_division=0)
    metrics['recall'] = recall_score(labels, predictions, zero_division=0)
    metrics['f1_score'] = f1_score(labels, predictions, zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(labels, predictions)
    metrics['confusion_matrix'] = cm
    
    # Additional derived metrics
    tn, fp, fn, tp = cm.ravel()
    metrics['true_negatives'] = int(tn)
    metrics['false_positives'] = int(fp)
    metrics['false_negatives'] = int(fn)
    metrics['true_positives'] = int(tp)
    
    # Specificity (True Negative Rate)
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # ROC-AUC
    fpr, tpr, _ = roc_curve(labels, probabilities)
    metrics['roc_auc'] = auc(fpr, tpr)
    metrics['fpr'] = fpr
    metrics['tpr'] = tpr
    
    return metrics


def print_metrics(metrics: Dict[str, float]):
    """
    Print evaluation metrics in a formatted manner.
    
    Args:
        metrics: Dictionary of computed metrics
    """
    print("\n" + "="*60)
    print("📊 EVALUATION METRICS")
    print("="*60)
    
    print("\n📈 Classification Metrics:")
    print("-" * 40)
    print(f"   Accuracy:    {metrics['accuracy']*100:.2f}%")
    print(f"   Precision:   {metrics['precision']*100:.2f}%")
    print(f"   Recall:      {metrics['recall']*100:.2f}%")
    print(f"   F1 Score:    {metrics['f1_score']*100:.2f}%")
    print(f"   Specificity: {metrics['specificity']*100:.2f}%")
    print(f"   ROC-AUC:     {metrics['roc_auc']*100:.2f}%")
    
    print("\n📊 Confusion Matrix Breakdown:")
    print("-" * 40)
    print(f"   True Positives (TP):  {metrics['true_positives']:5d}  (Fakes correctly detected)")
    print(f"   True Negatives (TN):  {metrics['true_negatives']:5d}  (Reals correctly identified)")
    print(f"   False Positives (FP): {metrics['false_positives']:5d}  (Reals misclassified as Fake)")
    print(f"   False Negatives (FN): {metrics['false_negatives']:5d}  (Fakes missed)")
    
    print("\n💡 Metric Interpretations:")
    print("-" * 40)
    
    # Interpret accuracy
    acc = metrics['accuracy'] * 100
    if acc >= 90:
        print(f"   ✅ Accuracy ({acc:.1f}%): Excellent overall performance")
    elif acc >= 75:
        print(f"   ⚠️ Accuracy ({acc:.1f}%): Good but room for improvement")
    else:
        print(f"   ❌ Accuracy ({acc:.1f}%): Model needs more training/data")
    
    # Interpret precision
    prec = metrics['precision'] * 100
    if prec >= 90:
        print(f"   ✅ Precision ({prec:.1f}%): Very few false alarms")
    elif prec >= 75:
        print(f"   ⚠️ Precision ({prec:.1f}%): Some real images flagged as fake")
    else:
        print(f"   ❌ Precision ({prec:.1f}%): Many false positives")
    
    # Interpret recall
    rec = metrics['recall'] * 100
    if rec >= 90:
        print(f"   ✅ Recall ({rec:.1f}%): Catching almost all fakes")
    elif rec >= 75:
        print(f"   ⚠️ Recall ({rec:.1f}%): Missing some deepfakes")
    else:
        print(f"   ❌ Recall ({rec:.1f}%): Many fakes slipping through")
    
    print()


def plot_confusion_matrix(
    cm: np.ndarray,
    save_path: str = 'confusion_matrix.png',
    show: bool = True
):
    """
    Plot and save the confusion matrix.
    
    Args:
        cm: Confusion matrix array
        save_path: Path to save the figure
        show: Whether to display the plot
    """
    plt.figure(figsize=(10, 8))
    
    # Create heatmap
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['Predicted Real', 'Predicted Fake'],
        yticklabels=['Actual Real', 'Actual Fake'],
        annot_kws={'size': 16},
        square=True
    )
    
    plt.title('Confusion Matrix - Deepfake Detection', fontsize=16, fontweight='bold')
    plt.ylabel('Actual Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    
    # Add text annotations explaining the quadrants
    plt.tight_layout()
    
    # Save figure
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Confusion matrix saved to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_roc_curve(
    fpr: np.ndarray,
    tpr: np.ndarray,
    roc_auc: float,
    save_path: str = 'roc_curve.png',
    show: bool = True
):
    """
    Plot and save the ROC curve.
    
    The ROC (Receiver Operating Characteristic) curve shows the tradeoff
    between True Positive Rate (Recall) and False Positive Rate at various
    classification thresholds.
    
    AUC (Area Under Curve):
    - 1.0: Perfect classifier
    - 0.5: Random classifier (diagonal line)
    - <0.5: Worse than random
    
    Args:
        fpr: False Positive Rates at different thresholds
        tpr: True Positive Rates at different thresholds
        roc_auc: Area under the ROC curve
        save_path: Path to save the figure
        show: Whether to display the plot
    """
    plt.figure(figsize=(10, 8))
    
    # Plot ROC curve
    plt.plot(
        fpr, tpr,
        color='darkorange',
        lw=2,
        label=f'ROC curve (AUC = {roc_auc:.3f})'
    )
    
    # Plot diagonal (random classifier)
    plt.plot(
        [0, 1], [0, 1],
        color='navy',
        lw=2,
        linestyle='--',
        label='Random classifier (AUC = 0.5)'
    )
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate (Recall)', fontsize=12)
    plt.title('ROC Curve - Deepfake Detection', fontsize=16, fontweight='bold')
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ ROC curve saved to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_training_history(
    history_path: str = 'training_history.json',
    save_path: str = 'training_curves.png',
    show: bool = True
):
    """
    Plot training and validation curves from saved history.
    
    Args:
        history_path: Path to training history JSON file
        save_path: Path to save the figure
        show: Whether to display the plot
    """
    if not os.path.exists(history_path):
        print(f"⚠️ Training history not found: {history_path}")
        return
    
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot Loss
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # Plot Accuracy
    axes[1].plot(epochs, history['train_acc'], 'b-', label='Training Accuracy', linewidth=2)
    axes[1].plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Training curves saved to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """
    Main function to run complete model evaluation.
    
    Steps:
    1. Load trained model
    2. Create test data loader
    3. Run inference on test set
    4. Compute metrics
    5. Generate visualizations
    6. Save results
    """
    print("\n" + "="*60)
    print("DEEPFAKE DETECTION - MODEL EVALUATION")
    print("="*60)
    
    # -------------------------------------------------------------------------
    # Configuration
    # -------------------------------------------------------------------------
    config = {
        'model_path': 'model.pth',
        'dataset_path': 'dataset',
        'batch_size': 32,
        'seed': 42,
    }
    
    # Set seed for reproducibility
    set_seed(config['seed'])
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️  Using device: {device}")
    
    # -------------------------------------------------------------------------
    # Check if model exists
    # -------------------------------------------------------------------------
    if not os.path.exists(config['model_path']):
        print(f"\n❌ Error: Model not found at '{config['model_path']}'")
        print("   Please run 'python train.py' first to train the model.")
        return
    
    # -------------------------------------------------------------------------
    # Load model
    # -------------------------------------------------------------------------
    model = load_model(config['model_path'], device)
    
    # -------------------------------------------------------------------------
    # Create test loader
    # -------------------------------------------------------------------------
    test_loader = get_test_loader(
        dataset_path=config['dataset_path'],
        batch_size=config['batch_size'],
        seed=config['seed']
    )
    
    # -------------------------------------------------------------------------
    # Evaluate model
    # -------------------------------------------------------------------------
    labels, predictions, probabilities = evaluate_model(model, test_loader, device)
    
    # -------------------------------------------------------------------------
    # Compute metrics
    # -------------------------------------------------------------------------
    metrics = compute_metrics(labels, predictions, probabilities)
    
    # Print metrics
    print_metrics(metrics)
    
    # -------------------------------------------------------------------------
    # Generate visualizations
    # -------------------------------------------------------------------------
    print("\n📊 Generating visualizations...")
    
    # Confusion Matrix
    plot_confusion_matrix(
        metrics['confusion_matrix'],
        save_path='confusion_matrix.png',
        show=False
    )
    
    # ROC Curve
    plot_roc_curve(
        metrics['fpr'],
        metrics['tpr'],
        metrics['roc_auc'],
        save_path='roc_curve.png',
        show=False
    )
    
    # Training history (if available)
    plot_training_history(
        history_path='training_history.json',
        save_path='training_curves.png',
        show=False
    )
    
    # -------------------------------------------------------------------------
    # Save metrics to JSON
    # -------------------------------------------------------------------------
    results = {
        'accuracy': float(metrics['accuracy']),
        'precision': float(metrics['precision']),
        'recall': float(metrics['recall']),
        'f1_score': float(metrics['f1_score']),
        'specificity': float(metrics['specificity']),
        'roc_auc': float(metrics['roc_auc']),
        'true_positives': metrics['true_positives'],
        'true_negatives': metrics['true_negatives'],
        'false_positives': metrics['false_positives'],
        'false_negatives': metrics['false_negatives'],
    }
    
    with open('evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ Evaluation results saved to: evaluation_results.json")
    
    # -------------------------------------------------------------------------
    # Print summary
    # -------------------------------------------------------------------------
    print("\n" + "="*60)
    print("✅ EVALUATION COMPLETE!")
    print("="*60)
    print("\nGenerated files:")
    print("   - confusion_matrix.png (Confusion matrix visualization)")
    print("   - roc_curve.png (ROC curve)")
    print("   - training_curves.png (Loss and accuracy curves)")
    print("   - evaluation_results.json (Metrics in JSON format)")
    print()


if __name__ == "__main__":
    main()
