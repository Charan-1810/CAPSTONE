"""
=============================================================================
MODEL.PY - Deepfake Detection CNN Model Architecture
=============================================================================

This module defines the CNN model for binary classification of images as
Real (0) or Fake (1).

Architecture Choice: ResNet18 (Pretrained on ImageNet)
------------------------------------------------------
WHY ResNet18?
1. Proven Performance: ResNet18 achieves excellent results on image tasks
2. Transfer Learning: Pretrained weights capture low-level features (edges,
   textures) that are useful for detecting manipulated images
3. Skip Connections: Residual connections help with gradient flow during
   fine-tuning, preventing vanishing gradients
4. Efficiency: Smaller than ResNet50/101, faster training while maintaining
   good accuracy - ideal for a baseline system
5. 224x224 Input: Matches standard ImageNet input size

Architecture Overview:
---------------------
Input: 224 × 224 × 3 (RGB image)
    ↓
[ResNet18 Backbone] - Pretrained convolutional layers
    ↓
Global Average Pooling → 512 features
    ↓
[Custom Classifier Head]
    - Linear(512, 256) + ReLU + Dropout(0.5)
    - Linear(256, 1) + Sigmoid
    ↓
Output: Single probability [0, 1] where 0=Real, 1=Fake
"""

import torch
import torch.nn as nn
from torchvision import models


class DeepfakeDetector(nn.Module):
    """
    Binary classifier for Deepfake Detection using pretrained ResNet18.
    
    The model uses transfer learning:
    1. Backbone (feature extractor): Frozen pretrained ResNet18 layers
    2. Classifier head: Custom fully connected layers for binary classification
    
    Attributes:
        backbone: ResNet18 without the final FC layer (feature extractor)
        classifier: Custom classification head
        
    Example:
        >>> model = DeepfakeDetector(pretrained=True, freeze_backbone=True)
        >>> input_tensor = torch.randn(1, 3, 224, 224)  # Batch of 1 image
        >>> output = model(input_tensor)  # Returns probability [0, 1]
    """
    
    def __init__(self, pretrained: bool = True, freeze_backbone: bool = True):
        """
        Initialize the Deepfake Detector model.
        
        Args:
            pretrained (bool): If True, use pretrained ImageNet weights.
                             This is crucial for transfer learning.
            freeze_backbone (bool): If True, freeze backbone weights initially.
                                   This prevents destroying pretrained features
                                   during early training when gradients are large.
        """
        super(DeepfakeDetector, self).__init__()
        
        # ---------------------------------------------------------------------
        # STEP 1: Load Pretrained ResNet18 Backbone
        # ---------------------------------------------------------------------
        # ResNet18 consists of:
        # - Initial conv layer (7x7, stride 2)
        # - Max pooling layer
        # - 4 residual blocks (layer1, layer2, layer3, layer4)
        # - Global average pooling
        # - Final fully connected layer (which we replace)
        
        if pretrained:
            # Use weights trained on ImageNet (1000 classes, millions of images)
            # These weights have learned powerful feature representations
            weights = models.ResNet18_Weights.IMAGENET1K_V1
            resnet = models.resnet18(weights=weights)
            print("✓ Loaded pretrained ResNet18 weights (ImageNet)")
        else:
            resnet = models.resnet18(weights=None)
            print("✓ Initialized ResNet18 with random weights")
        
        # ---------------------------------------------------------------------
        # STEP 2: Extract Feature Extractor (Remove Final FC Layer)
        # ---------------------------------------------------------------------
        # nn.Sequential(*list(resnet.children())[:-1]) takes all layers EXCEPT
        # the last one (the 1000-class classifier)
        # This gives us a feature extractor that outputs 512-dimensional vectors
        
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        # Number of features output by ResNet18's backbone
        self.num_features = resnet.fc.in_features  # 512 for ResNet18
        
        # ---------------------------------------------------------------------
        # STEP 3: Freeze Backbone (Optional but Recommended Initially)
        # ---------------------------------------------------------------------
        # WHY FREEZE?
        # - Pretrained weights are valuable and shouldn't be disturbed early
        # - Small datasets can cause overfitting if all weights are trainable
        # - Training only the classifier is faster and more stable
        # 
        # Later, we can unfreeze (fine-tune) for better performance
        
        if freeze_backbone:
            self._freeze_backbone()
            print("✓ Backbone frozen - only classifier will be trained")
        else:
            print("✓ Backbone unfrozen - all layers will be trained")
        
        # ---------------------------------------------------------------------
        # STEP 4: Custom Classification Head
        # ---------------------------------------------------------------------
        # This replaces ResNet's 1000-class output with binary classification
        #
        # Architecture:
        # - Linear(512 → 256): Reduces dimensionality, learns task-specific features
        # - ReLU: Non-linearity for learning complex patterns
        # - Dropout(0.5): Regularization to prevent overfitting (50% dropout)
        # - Linear(256 → 1): Final binary classification
        # - Sigmoid: Converts output to probability [0, 1]
        
        self.classifier = nn.Sequential(
            nn.Linear(self.num_features, 256),  # 512 → 256
            nn.ReLU(inplace=True),               # Activation function
            nn.Dropout(p=0.5),                   # Regularization
            nn.Linear(256, 1),                   # 256 → 1 (binary output)
            nn.Sigmoid()                         # Output probability [0, 1]
        )
        
        print(f"✓ Classifier head created: {self.num_features} → 256 → 1")
    
    def _freeze_backbone(self):
        """
        Freeze all backbone parameters to prevent weight updates.
        
        When a parameter has requires_grad=False:
        - Gradients are not computed for it during backpropagation
        - Its values remain unchanged during optimizer.step()
        """
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self):
        """
        Unfreeze backbone for fine-tuning.
        
        Call this after initial training on the classifier head to allow
        the model to adapt pretrained features to the deepfake detection task.
        
        Fine-tuning typically uses a lower learning rate (e.g., 1e-5) to
        make small adjustments without destroying learned features.
        """
        for param in self.backbone.parameters():
            param.requires_grad = True
        print("✓ Backbone unfrozen - all layers will now be trained")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, 224, 224)
                            - 3 channels (RGB)
                            - 224x224 spatial dimensions
        
        Returns:
            torch.Tensor: Predictions of shape (batch_size, 1)
                         Values are probabilities in range [0, 1]
                         - Close to 0 → Real image
                         - Close to 1 → Fake image
        
        Processing Steps:
            1. Input: (B, 3, 224, 224)
            2. After backbone: (B, 512, 1, 1)  # Global avg pooling included
            3. After flatten: (B, 512)
            4. After classifier: (B, 1)
        """
        # Extract features using backbone
        # Output shape: (batch_size, 512, 1, 1)
        features = self.backbone(x)
        
        # Flatten the features
        # Output shape: (batch_size, 512)
        features = features.view(features.size(0), -1)
        
        # Pass through classifier
        # Output shape: (batch_size, 1)
        output = self.classifier(features)
        
        return output
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features before the classifier (useful for Grad-CAM).
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, 224, 224)
        
        Returns:
            torch.Tensor: Feature tensor of shape (batch_size, 512)
        """
        features = self.backbone(x)
        features = features.view(features.size(0), -1)
        return features


def get_model(pretrained: bool = True, freeze_backbone: bool = True) -> DeepfakeDetector:
    """
    Factory function to create a DeepfakeDetector model.
    
    Args:
        pretrained (bool): Use pretrained ImageNet weights
        freeze_backbone (bool): Freeze backbone initially
    
    Returns:
        DeepfakeDetector: Initialized model
    
    Example:
        >>> model = get_model(pretrained=True, freeze_backbone=True)
        >>> model = model.to('cuda')  # Move to GPU
    """
    model = DeepfakeDetector(pretrained=pretrained, freeze_backbone=freeze_backbone)
    return model


def count_parameters(model: nn.Module) -> tuple:
    """
    Count trainable and total parameters in a model.
    
    Args:
        model (nn.Module): PyTorch model
    
    Returns:
        tuple: (trainable_params, total_params)
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


# =============================================================================
# MODULE TEST
# =============================================================================
if __name__ == "__main__":
    """
    Test the model architecture when running this file directly.
    
    Run: python model.py
    """
    print("\n" + "="*60)
    print("TESTING DEEPFAKE DETECTOR MODEL")
    print("="*60 + "\n")
    
    # Create model
    model = get_model(pretrained=True, freeze_backbone=True)
    
    # Count parameters
    trainable, total = count_parameters(model)
    print(f"\n📊 Model Statistics:")
    print(f"   - Total parameters: {total:,}")
    print(f"   - Trainable parameters: {trainable:,}")
    print(f"   - Frozen parameters: {total - trainable:,}")
    
    # Test forward pass with dummy input
    print(f"\n🧪 Testing forward pass...")
    dummy_input = torch.randn(4, 3, 224, 224)  # Batch of 4 images
    output = model(dummy_input)
    
    print(f"   - Input shape: {dummy_input.shape}")
    print(f"   - Output shape: {output.shape}")
    print(f"   - Output values (probabilities): {output.squeeze().tolist()}")
    
    # Test with backbone unfrozen
    print(f"\n🔓 Testing backbone unfreeze...")
    model.unfreeze_backbone()
    trainable_after, _ = count_parameters(model)
    print(f"   - Trainable parameters after unfreeze: {trainable_after:,}")
    
    print("\n✅ Model test complete!\n")
