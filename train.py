"""
=============================================================================
TRAIN.PY - Complete Training Pipeline for Deepfake Detection
=============================================================================

This module implements the complete training pipeline including:
1. Data loading and preprocessing
2. Dataset creation with augmentation
3. Stratified train/val/test split
4. Training loop with validation
5. Model checkpointing

PREPROCESSING EXPLAINED:
-----------------------

1. RESIZE TO 224×224:
   - ResNet18 expects 224×224 input (ImageNet standard)
   - Ensures consistent input dimensions across all images
   - Maintains aspect ratio compatibility with pretrained weights

2. NORMALIZE WITH IMAGENET STATS:
   - mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
   - These are RGB channel statistics calculated from ImageNet dataset
   - Using same normalization as pretraining ensures feature consistency
   - Helps the pretrained backbone work optimally

3. DATA AUGMENTATION (Training Only):
   - RandomHorizontalFlip: Deepfakes can appear flipped; adds invariance
   - RandomRotation(10): Small rotations simulate real-world variations
   - ColorJitter(brightness=0.2): Handles lighting variations in images
   - WHY? Augmentation artificially increases dataset diversity,
     reducing overfitting and improving generalization

4. STRATIFIED SPLIT (80/10/10):
   - Stratified: Maintains class balance (real/fake ratio) in each split
   - 80% train: Main learning data
   - 10% validation: Hyperparameter tuning, early stopping
   - 10% test: Final unbiased performance evaluation
"""

import os
import copy
import random
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Tuple, Dict, List, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from model import get_model, count_parameters


# =============================================================================
# REPRODUCIBILITY - Set random seeds for consistent results
# =============================================================================

def set_seed(seed: int = 42):
    """
    Set random seeds for reproducibility.
    
    WHY IS THIS IMPORTANT?
    - Deep learning involves many random operations (weight init, shuffling, etc.)
    - Setting seeds ensures experiments can be reproduced exactly
    - Makes debugging easier since results are consistent
    
    Args:
        seed (int): Random seed value (default: 42)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"✓ Random seed set to {seed}")


# =============================================================================
# CUSTOM DATASET CLASS
# =============================================================================

class DeepfakeDataset(Dataset):
    """
    Custom PyTorch Dataset for loading deepfake detection images.
    
    Expected folder structure:
        dataset/
            real/
                image1.jpg
                image2.png
                ...
            fake/
                image1.jpg
                image2.png
                ...
    
    Labels:
        - real = 0
        - fake = 1
    
    Attributes:
        image_paths (List[str]): List of all image file paths
        labels (List[int]): Corresponding labels (0=real, 1=fake)
        transform: Torchvision transforms to apply to images
    """
    
    SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    
    def __init__(self, data_dir: str, transform=None):
        """
        Initialize the dataset.
        
        Args:
            data_dir (str): Path to the dataset directory containing 
                           'real' and 'fake' subfolders
            transform: Torchvision transforms to apply (default: None)
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        
        # Validate directory structure
        self._validate_directory()
        
        # Collect all image paths and labels
        self.image_paths = []
        self.labels = []
        self._load_dataset()
        
        print(f"✓ Dataset loaded: {len(self.image_paths)} images")
        print(f"  - Real images: {self.labels.count(0)}")
        print(f"  - Fake images: {self.labels.count(1)}")
    
    def _validate_directory(self):
        """Validate that the dataset directory has the expected structure."""
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {self.data_dir}")
        
        real_dir = self.data_dir / 'real'
        fake_dir = self.data_dir / 'fake'
        
        if not real_dir.exists():
            raise FileNotFoundError(f"'real' folder not found in {self.data_dir}")
        if not fake_dir.exists():
            raise FileNotFoundError(f"'fake' folder not found in {self.data_dir}")
    
    def _load_dataset(self):
        """Load all image paths and their corresponding labels."""
        # Load real images (label = 0)
        real_dir = self.data_dir / 'real'
        for img_path in real_dir.iterdir():
            if img_path.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                self.image_paths.append(str(img_path))
                self.labels.append(0)  # Real = 0
        
        # Load fake images (label = 1)
        fake_dir = self.data_dir / 'fake'
        for img_path in fake_dir.iterdir():
            if img_path.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                self.image_paths.append(str(img_path))
                self.labels.append(1)  # Fake = 1
    
    def __len__(self) -> int:
        """Return the total number of samples in the dataset."""
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a single sample from the dataset.
        
        Args:
            idx (int): Index of the sample
        
        Returns:
            tuple: (image_tensor, label)
                - image_tensor: Transformed image tensor (3, 224, 224)
                - label: Integer label (0=real, 1=fake)
        """
        # Load image
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')  # Ensure RGB format
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Get label
        label = self.labels[idx]
        
        return image, label


# =============================================================================
# DATA TRANSFORMS (Preprocessing + Augmentation)
# =============================================================================

def get_transforms(is_training: bool = True) -> transforms.Compose:
    """
    Create image transforms for training or evaluation.
    
    WHY THESE SPECIFIC TRANSFORMS?
    
    TRAINING TRANSFORMS:
    1. Resize(256): Slightly larger than final size for random cropping
    2. RandomResizedCrop(224): Adds scale variation, randomly crops to 224
    3. RandomHorizontalFlip: 50% chance to flip - adds horizontal invariance
    4. RandomRotation(10): ±10° rotation - simulates camera angle variation
    5. ColorJitter: Brightness variation ±20% - handles lighting changes
    6. ToTensor: Converts PIL Image to PyTorch tensor (0-255 → 0-1)
    7. Normalize: Standardizes using ImageNet statistics
    
    EVALUATION TRANSFORMS:
    - Only resize, center crop, and normalize
    - No augmentation (we want consistent evaluation)
    
    Args:
        is_training (bool): If True, include data augmentation
    
    Returns:
        transforms.Compose: Composition of transforms
    """
    # ImageNet normalization statistics
    # These values are computed from millions of ImageNet images
    imagenet_mean = [0.485, 0.456, 0.406]  # RGB means
    imagenet_std = [0.229, 0.224, 0.225]   # RGB standard deviations
    
    if is_training:
        # Training: Include augmentation for better generalization
        return transforms.Compose([
            transforms.Resize(256),                    # Resize to 256 for cropping
            transforms.RandomResizedCrop(224),        # Random crop to 224
            transforms.RandomHorizontalFlip(p=0.5),   # 50% horizontal flip
            transforms.RandomRotation(degrees=10),    # ±10° rotation
            transforms.ColorJitter(brightness=0.2),   # ±20% brightness
            transforms.ToTensor(),                    # Convert to tensor
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std)  # Normalize
        ])
    else:
        # Evaluation: No augmentation, deterministic processing
        return transforms.Compose([
            transforms.Resize(256),                   # Resize to 256
            transforms.CenterCrop(224),               # Center crop to 224
            transforms.ToTensor(),                    # Convert to tensor
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std)  # Normalize
        ])


# =============================================================================
# STRATIFIED DATA SPLIT
# =============================================================================

def create_data_loaders(
    dataset_path: str,
    batch_size: int = 32,
    num_workers: int = 4,
    val_split: float = 0.1,
    test_split: float = 0.1,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test data loaders with stratified split.
    
    STRATIFIED SPLIT EXPLAINED:
    - Ensures each split (train/val/test) has same class distribution
    - Example: If dataset is 60% fake, 40% real, each split will be ~60/40
    - Prevents bias from uneven splits (e.g., all fakes in training)
    - Critical for reliable model evaluation
    
    Args:
        dataset_path (str): Path to dataset directory
        batch_size (int): Number of samples per batch (default: 32)
        num_workers (int): Number of parallel data loading workers
        val_split (float): Fraction for validation (default: 0.1 = 10%)
        test_split (float): Fraction for testing (default: 0.1 = 10%)
        seed (int): Random seed for reproducibility
    
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    print("\n" + "="*60)
    print("CREATING DATA LOADERS")
    print("="*60)
    
    # Create full dataset with evaluation transforms (just for getting indices)
    full_dataset = DeepfakeDataset(dataset_path, transform=None)
    labels = full_dataset.labels
    indices = list(range(len(full_dataset)))
    
    # ---------------------------------------------------------------------
    # STEP 1: Split into train+val and test (stratified)
    # ---------------------------------------------------------------------
    train_val_indices, test_indices = train_test_split(
        indices,
        test_size=test_split,
        stratify=labels,  # Maintain class balance
        random_state=seed
    )
    
    # Get labels for train_val subset
    train_val_labels = [labels[i] for i in train_val_indices]
    
    # ---------------------------------------------------------------------
    # STEP 2: Split train+val into train and val (stratified)
    # ---------------------------------------------------------------------
    # Adjust val_split to account for already removed test set
    adjusted_val_split = val_split / (1 - test_split)
    
    train_indices, val_indices = train_test_split(
        train_val_indices,
        test_size=adjusted_val_split,
        stratify=train_val_labels,
        random_state=seed
    )
    
    # ---------------------------------------------------------------------
    # STEP 3: Create datasets with appropriate transforms
    # ---------------------------------------------------------------------
    # Training dataset: with augmentation
    train_dataset = DeepfakeDataset(dataset_path, transform=get_transforms(is_training=True))
    train_subset = Subset(train_dataset, train_indices)
    
    # Validation dataset: without augmentation
    val_dataset = DeepfakeDataset(dataset_path, transform=get_transforms(is_training=False))
    val_subset = Subset(val_dataset, val_indices)
    
    # Test dataset: without augmentation
    test_dataset = DeepfakeDataset(dataset_path, transform=get_transforms(is_training=False))
    test_subset = Subset(test_dataset, test_indices)
    
    # ---------------------------------------------------------------------
    # STEP 4: Create DataLoaders
    # ---------------------------------------------------------------------
    # DataLoader handles batching, shuffling, and parallel loading
    
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,          # Shuffle training data each epoch
        num_workers=num_workers,
        pin_memory=True        # Faster GPU transfer
    )
    
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,         # No need to shuffle validation
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_subset,
        batch_size=batch_size,
        shuffle=False,         # No need to shuffle test
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Print split statistics
    print(f"\n📊 Data Split Statistics:")
    print(f"   - Total samples: {len(full_dataset)}")
    print(f"   - Training samples: {len(train_indices)} ({len(train_indices)/len(full_dataset)*100:.1f}%)")
    print(f"   - Validation samples: {len(val_indices)} ({len(val_indices)/len(full_dataset)*100:.1f}%)")
    print(f"   - Test samples: {len(test_indices)} ({len(test_indices)/len(full_dataset)*100:.1f}%)")
    print(f"   - Batch size: {batch_size}")
    print(f"   - Training batches: {len(train_loader)}")
    
    return train_loader, val_loader, test_loader


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int
) -> Tuple[float, float]:
    """
    Train the model for one epoch.
    
    Args:
        model: Neural network model
        train_loader: Training data loader
        criterion: Loss function (BCELoss)
        optimizer: Optimizer (Adam)
        device: Device to train on (cuda/cpu)
        epoch: Current epoch number
    
    Returns:
        tuple: (average_loss, accuracy)
    """
    model.train()  # Set model to training mode (enables dropout, etc.)
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Progress bar
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]", leave=False)
    
    for batch_idx, (images, labels) in enumerate(pbar):
        # ---------------------------------------------------------------------
        # STEP 1: Move data to device
        # ---------------------------------------------------------------------
        images = images.to(device)
        labels = labels.float().unsqueeze(1).to(device)  # Shape: (batch, 1)
        
        # ---------------------------------------------------------------------
        # STEP 2: Forward pass
        # ---------------------------------------------------------------------
        optimizer.zero_grad()  # Clear previous gradients
        outputs = model(images)  # Get predictions
        
        # ---------------------------------------------------------------------
        # STEP 3: Compute loss
        # ---------------------------------------------------------------------
        loss = criterion(outputs, labels)
        
        # ---------------------------------------------------------------------
        # STEP 4: Backward pass (compute gradients)
        # ---------------------------------------------------------------------
        loss.backward()
        
        # ---------------------------------------------------------------------
        # STEP 5: Update weights
        # ---------------------------------------------------------------------
        optimizer.step()
        
        # ---------------------------------------------------------------------
        # STEP 6: Track metrics
        # ---------------------------------------------------------------------
        running_loss += loss.item()
        
        # Convert probabilities to binary predictions (threshold = 0.5)
        predicted = (outputs > 0.5).float()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100*correct/total:.2f}%'
        })
    
    # Calculate epoch metrics
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float]:
    """
    Validate the model on the validation set.
    
    No gradient computation (faster and uses less memory).
    
    Args:
        model: Neural network model
        val_loader: Validation data loader
        criterion: Loss function
        device: Device to validate on
    
    Returns:
        tuple: (average_loss, accuracy)
    """
    model.eval()  # Set model to evaluation mode (disables dropout, etc.)
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():  # Disable gradient computation
        pbar = tqdm(val_loader, desc="Validating", leave=False)
        
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.float().unsqueeze(1).to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    val_loss = running_loss / len(val_loader)
    val_acc = 100 * correct / total
    
    return val_loss, val_acc


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Dict,
    device: torch.device
) -> Tuple[nn.Module, Dict]:
    """
    Complete training loop with validation and model checkpointing.
    
    TRAINING STRATEGY:
    1. Phase 1 (Epochs 1-5): Train only classifier head (backbone frozen)
       - Prevents destruction of pretrained features
       - Fast convergence for classification layer
    
    2. Phase 2 (Epochs 6+): Fine-tune entire model (backbone unfrozen)
       - Adapts pretrained features to deepfake detection task
       - Uses lower learning rate to prevent catastrophic forgetting
    
    Args:
        model: Neural network model
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Training configuration dictionary
        device: Device to train on
    
    Returns:
        tuple: (best_model, training_history)
    """
    print("\n" + "="*60)
    print("TRAINING MODEL")
    print("="*60)
    
    # ---------------------------------------------------------------------
    # SETUP: Loss function, optimizer, and scheduler
    # ---------------------------------------------------------------------
    
    # Binary Cross Entropy Loss with class weights
    # WHY BCE? Perfect for binary classification with sigmoid output
    # Measures difference between predicted probability and true label
    # We use BCEWithLogitsLoss for better numerical stability, but since
    # our model already has sigmoid, we use pos_weight with BCELoss manually
    criterion = nn.BCELoss()
    
    # Note: If class imbalance exists, consider using weighted loss
    # For now, we'll handle this during training by monitoring both classes
    
    # Adam Optimizer
    # WHY ADAM? Adaptive learning rate, works well out-of-the-box
    # Combines benefits of AdaGrad and RMSProp
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config.get('weight_decay', 1e-5)  # L2 regularization
    )
    
    # Learning rate scheduler (optional)
    # Reduces LR when validation loss plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=2
    )
    
    # ---------------------------------------------------------------------
    # TRAINING LOOP
    # ---------------------------------------------------------------------
    
    best_val_loss = float('inf')
    best_model_weights = copy.deepcopy(model.state_dict())
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    epochs = config['epochs']
    unfreeze_epoch = config.get('unfreeze_epoch', 5)
    
    print(f"\n📋 Training Configuration:")
    print(f"   - Epochs: {epochs}")
    print(f"   - Learning rate: {config['learning_rate']}")
    print(f"   - Device: {device}")
    print(f"   - Backbone unfreezing at epoch: {unfreeze_epoch}")
    print()
    
    for epoch in range(epochs):
        # Unfreeze backbone for fine-tuning after initial epochs
        if epoch == unfreeze_epoch:
            print(f"\n🔓 Unfreezing backbone for fine-tuning...")
            model.unfreeze_backbone()
            
            # Optionally reduce learning rate for fine-tuning
            for param_group in optimizer.param_groups:
                param_group['lr'] = config['learning_rate'] / 10
            print(f"   Learning rate reduced to {config['learning_rate']/10}")
        
        # Train and validate
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Record history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Print epoch results
        print(f"Epoch {epoch+1}/{epochs}:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_weights = copy.deepcopy(model.state_dict())
            torch.save(best_model_weights, config['model_save_path'])
            print(f"  ✓ Best model saved! (Val Loss: {val_loss:.4f})")
        
        print()
    
    # Load best model weights
    model.load_state_dict(best_model_weights)
    
    print("="*60)
    print(f"✅ Training complete! Best validation loss: {best_val_loss:.4f}")
    print(f"✅ Model saved to: {config['model_save_path']}")
    print("="*60)
    
    return model, history


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """
    Main function to run the complete training pipeline.
    
    Steps:
    1. Set random seeds for reproducibility
    2. Configure training parameters
    3. Create data loaders
    4. Initialize model
    5. Train model
    6. Save final model and training history
    """
    print("\n" + "="*60)
    print("DEEPFAKE DETECTION - TRAINING PIPELINE")
    print("="*60)
    
    # -------------------------------------------------------------------------
    # STEP 1: Set random seeds for reproducibility
    # -------------------------------------------------------------------------
    set_seed(42)
    
    # -------------------------------------------------------------------------
    # STEP 2: Configuration
    # -------------------------------------------------------------------------
    config = {
        # Data parameters
        'dataset_path': 'dataset',           # Path to dataset directory
        'batch_size': 32,                    # Number of samples per batch
        'num_workers': 4,                    # Parallel data loading workers
        
        # Training parameters
        'epochs': 10,                        # Total training epochs
        'learning_rate': 1e-4,               # Initial learning rate
        'weight_decay': 1e-5,                # L2 regularization strength
        'unfreeze_epoch': 5,                 # When to unfreeze backbone
        
        # Model parameters
        'pretrained': True,                  # Use pretrained weights
        'freeze_backbone': True,             # Freeze backbone initially
        
        # Output paths
        'model_save_path': 'model.pth',      # Where to save best model
    }
    
    print("\n📋 Configuration:")
    for key, value in config.items():
        print(f"   {key}: {value}")
    
    # -------------------------------------------------------------------------
    # STEP 3: Setup device (GPU if available)
    # -------------------------------------------------------------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️  Using device: {device}")
    if device.type == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # -------------------------------------------------------------------------
    # STEP 4: Create data loaders
    # -------------------------------------------------------------------------
    train_loader, val_loader, test_loader = create_data_loaders(
        dataset_path=config['dataset_path'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers']
    )
    
    # -------------------------------------------------------------------------
    # STEP 5: Initialize model
    # -------------------------------------------------------------------------
    print("\n" + "="*60)
    print("INITIALIZING MODEL")
    print("="*60)
    
    model = get_model(
        pretrained=config['pretrained'],
        freeze_backbone=config['freeze_backbone']
    )
    model = model.to(device)
    
    trainable, total = count_parameters(model)
    print(f"\n📊 Model Parameters:")
    print(f"   - Total: {total:,}")
    print(f"   - Trainable: {trainable:,}")
    
    # -------------------------------------------------------------------------
    # STEP 6: Train model
    # -------------------------------------------------------------------------
    model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device
    )
    
    # -------------------------------------------------------------------------
    # STEP 7: Save training history
    # -------------------------------------------------------------------------
    import json
    with open('training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    print(f"✓ Training history saved to training_history.json")
    
    print("\n" + "="*60)
    print("🎉 TRAINING PIPELINE COMPLETE!")
    print("="*60)
    print("\nNext steps:")
    print("  1. Run 'python evaluate.py' to evaluate on test set")
    print("  2. Run 'streamlit run app.py' to launch web interface")
    print()


if __name__ == "__main__":
    main()
