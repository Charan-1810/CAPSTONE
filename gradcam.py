"""
=============================================================================
GRADCAM.PY - Gradient-weighted Class Activation Mapping for Explainability
=============================================================================

This module implements Grad-CAM (Gradient-weighted Class Activation Mapping)
to visualize which regions of an image the model focuses on when making
predictions about whether an image is real or fake.

WHAT IS GRAD-CAM?
-----------------
Grad-CAM is an explainability technique that produces visual explanations
for decisions from CNN-based models. It highlights the important regions
in an image that contributed to a specific prediction.

HOW DOES GRAD-CAM WORK?
----------------------

1. FORWARD PASS:
   - Input image passes through the CNN
   - We capture the feature maps (activations) from the last convolutional layer
   - These feature maps contain spatial information about "what" and "where"

2. BACKWARD PASS:
   - Compute gradients of the model's output with respect to feature maps
   - These gradients tell us how much each feature map affects the output

3. GLOBAL AVERAGE POOLING:
   - Average the gradients over spatial dimensions (height × width)
   - This gives us importance weights (α) for each feature map channel

4. WEIGHTED COMBINATION:
   - Multiply each feature map by its importance weight
   - Sum all weighted feature maps
   - Apply ReLU (keep only positive influence)

5. VISUALIZATION:
   - Upsample the heatmap to input image size
   - Normalize values to [0, 1]
   - Apply colormap (e.g., jet) for visualization
   - Overlay on original image

WHY IS GRAD-CAM USEFUL FOR DEEPFAKE DETECTION?
----------------------------------------------

1. INTERPRETABILITY:
   - Shows which facial regions the model considers "fake" indicators
   - Common manipulation artifacts: eyes, mouth, face boundaries, hair edges

2. DEBUGGING:
   - If model focuses on background, it may be learning wrong features
   - Helps identify dataset biases

3. TRUST:
   - Users can verify the model's reasoning
   - Critical for real-world deployment

4. RESEARCH:
   - Understand what distinguishes real from fake images
   - Design better models based on insights

MATHEMATICAL FORMULATION:
------------------------

L^c_Grad-CAM = ReLU(∑_k α^c_k · A^k)

Where:
- L^c_Grad-CAM: The Grad-CAM heatmap for class c
- α^c_k: Importance weight for feature map k (for class c)
- A^k: Feature map k from the convolutional layer
- ReLU: Rectified Linear Unit (removes negative values)

α^c_k = (1/Z) ∑_i ∑_j ∂y^c / ∂A^k_ij

Where:
- Z: Number of pixels in the feature map
- ∂y^c / ∂A^k_ij: Gradient of class score with respect to feature map activation
"""

import numpy as np
from PIL import Image
from typing import Tuple, Optional
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from model import DeepfakeDetector


class GradCAM:
    """
    Grad-CAM implementation for DeepfakeDetector model.
    
    This class handles:
    1. Hook registration to capture activations and gradients
    2. Forward and backward passes
    3. Heatmap computation
    4. Visualization generation
    
    Attributes:
        model: The DeepfakeDetector model
        target_layer: The layer to compute Grad-CAM from (usually last conv layer)
        activations: Stored feature maps from forward pass
        gradients: Stored gradients from backward pass
    
    Example:
        >>> gradcam = GradCAM(model, target_layer=model.backbone[-2])
        >>> heatmap = gradcam.generate(input_tensor)
        >>> overlay = gradcam.overlay_heatmap(image, heatmap)
    """
    
    def __init__(self, model: DeepfakeDetector, target_layer: nn.Module):
        """
        Initialize Grad-CAM.
        
        Args:
            model: Trained DeepfakeDetector model
            target_layer: Layer to compute Grad-CAM from
                         For ResNet18, this is typically the last residual block
        """
        self.model = model
        self.model.eval()  # Set to evaluation mode
        
        self.target_layer = target_layer
        
        # Storage for activations and gradients
        self.activations = None
        self.gradients = None
        
        # Register hooks
        self._register_hooks()
    
    def _register_hooks(self):
        """
        Register forward and backward hooks on target layer.
        
        HOOKS EXPLAINED:
        - Forward hook: Called during forward pass, captures activations
        - Backward hook: Called during backward pass, captures gradients
        
        Both are needed to compute the weighted combination in Grad-CAM.
        """
        
        def forward_hook(module, input, output):
            """Store activations during forward pass."""
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            """Store gradients during backward pass."""
            self.gradients = grad_output[0].detach()
        
        # Register hooks
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)
    
    def generate(self, input_tensor: torch.Tensor) -> np.ndarray:
        """
        Generate Grad-CAM heatmap for the input image.
        
        Args:
            input_tensor: Preprocessed image tensor (1, 3, 224, 224)
        
        Returns:
            np.ndarray: Heatmap of shape (H, W) with values in [0, 1]
        
        Steps:
            1. Forward pass to get prediction and activations
            2. Backward pass to get gradients
            3. Compute importance weights (global average pool of gradients)
            4. Weighted combination of feature maps
            5. Apply ReLU and normalize
        """
        # Ensure model is in eval mode
        self.model.eval()
        
        # Enable gradients for input (needed for backward pass)
        input_tensor = input_tensor.requires_grad_(True)
        
        # ---------------------------------------------------------------------
        # STEP 1: Forward pass
        # ---------------------------------------------------------------------
        # This triggers the forward hook, storing activations
        output = self.model(input_tensor)
        
        # For binary classification with sigmoid, use the output directly
        # (higher output = more "fake")
        target = output
        
        # ---------------------------------------------------------------------
        # STEP 2: Backward pass
        # ---------------------------------------------------------------------
        # Zero existing gradients
        self.model.zero_grad()
        
        # Compute gradients
        # This triggers the backward hook, storing gradients
        target.backward(retain_graph=True)
        
        # ---------------------------------------------------------------------
        # STEP 3: Compute importance weights
        # ---------------------------------------------------------------------
        # Global average pooling of gradients
        # Shape: (batch, channels, height, width) → (batch, channels)
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        
        # ---------------------------------------------------------------------
        # STEP 4: Weighted combination of feature maps
        # ---------------------------------------------------------------------
        # Multiply activations by weights and sum across channels
        # Shape: (batch, channels, height, width) → (batch, 1, height, width)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        
        # ---------------------------------------------------------------------
        # STEP 5: Apply ReLU (keep only positive influence)
        # ---------------------------------------------------------------------
        cam = F.relu(cam)
        
        # ---------------------------------------------------------------------
        # STEP 6: Resize to input size and normalize
        # ---------------------------------------------------------------------
        # Resize to input image dimensions
        cam = F.interpolate(
            cam,
            size=(224, 224),
            mode='bilinear',
            align_corners=False
        )
        
        # Convert to numpy and normalize to [0, 1]
        cam = cam.squeeze().cpu().numpy()
        
        # Normalize (avoid division by zero)
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)
        
        return cam
    
    @staticmethod
    def apply_colormap(heatmap: np.ndarray, colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
        """
        Apply colormap to grayscale heatmap.
        
        The JET colormap provides intuitive visualization:
        - Blue: Low activation (less important)
        - Green/Yellow: Medium activation
        - Red: High activation (most important)
        
        Args:
            heatmap: Grayscale heatmap (H, W) with values in [0, 1]
            colormap: OpenCV colormap (default: JET)
        
        Returns:
            np.ndarray: RGB heatmap (H, W, 3)
        """
        # Scale to 0-255 and convert to uint8
        heatmap_uint8 = (heatmap * 255).astype(np.uint8)
        
        # Apply colormap
        colored = cv2.applyColorMap(heatmap_uint8, colormap)
        
        # Convert BGR to RGB
        colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
        
        return colored
    
    @staticmethod
    def overlay_heatmap(
        original_image: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.5,
        colormap: int = cv2.COLORMAP_JET
    ) -> np.ndarray:
        """
        Overlay heatmap on original image.
        
        Args:
            original_image: Original RGB image (H, W, 3)
            heatmap: Grayscale heatmap (H, W) with values in [0, 1]
            alpha: Transparency of heatmap overlay (0=invisible, 1=opaque)
            colormap: OpenCV colormap
        
        Returns:
            np.ndarray: RGB image with heatmap overlay
        """
        # Ensure original image is right size
        original_image = np.array(Image.fromarray(original_image).resize((224, 224)))
        
        # Apply colormap to heatmap
        colored_heatmap = GradCAM.apply_colormap(heatmap, colormap)
        
        # Blend original and heatmap
        overlay = (
            (1 - alpha) * original_image.astype(np.float32) +
            alpha * colored_heatmap.astype(np.float32)
        )
        
        # Clip to valid range and convert to uint8
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)
        
        return overlay


def get_gradcam(model: DeepfakeDetector) -> GradCAM:
    """
    Create GradCAM instance for DeepfakeDetector.
    
    Uses the last convolutional layer of ResNet18's backbone for
    computing Grad-CAM. This layer (layer4) captures high-level
    features that are most relevant for classification.
    
    Args:
        model: Trained DeepfakeDetector model
    
    Returns:
        GradCAM: Initialized GradCAM instance
    """
    # For ResNet18, the backbone structure is:
    # backbone[0]: Conv1
    # backbone[1]: BatchNorm
    # backbone[2]: ReLU
    # backbone[3]: MaxPool
    # backbone[4]: layer1 (2 residual blocks)
    # backbone[5]: layer2 (2 residual blocks)
    # backbone[6]: layer3 (2 residual blocks)
    # backbone[7]: layer4 (2 residual blocks) ← We use this
    # backbone[8]: AdaptiveAvgPool
    
    # Get the last convolutional layer (layer4)
    target_layer = model.backbone[7]
    
    return GradCAM(model, target_layer)


def preprocess_image(image: Image.Image) -> Tuple[torch.Tensor, np.ndarray]:
    """
    Preprocess image for model input and Grad-CAM visualization.
    
    Args:
        image: PIL Image (RGB)
    
    Returns:
        tuple: (preprocessed_tensor, original_array)
            - preprocessed_tensor: Normalized tensor (1, 3, 224, 224)
            - original_array: Original image as numpy array (224, 224, 3)
    """
    # ImageNet normalization
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Preprocess for model
    tensor = transform(image).unsqueeze(0)  # Add batch dimension
    
    # Keep original for visualization (resize to 224x224)
    original = np.array(image.resize((224, 224)))
    
    return tensor, original


def generate_gradcam_visualization(
    model: DeepfakeDetector,
    image: Image.Image,
    device: torch.device = torch.device('cpu')
) -> Tuple[np.ndarray, np.ndarray, float, str]:
    """
    Generate complete Grad-CAM visualization for an image.
    
    This is the main function to use for generating explanations.
    
    Args:
        model: Trained DeepfakeDetector model
        image: PIL Image to analyze
        device: Device for computation
    
    Returns:
        tuple: (overlay, heatmap, confidence, prediction)
            - overlay: Original image with heatmap overlay (H, W, 3)
            - heatmap: Colored heatmap (H, W, 3)
            - confidence: Prediction confidence (0-100)
            - prediction: "REAL" or "FAKE"
    """
    # Move model to device
    model = model.to(device)
    model.eval()
    
    # Preprocess image
    input_tensor, original_image = preprocess_image(image)
    input_tensor = input_tensor.to(device)
    
    # Get prediction
    with torch.no_grad():
        output = model(input_tensor)
        prob_fake = output.item()  # Probability of being fake
    
    # Determine prediction
    if prob_fake > 0.5:
        prediction = "FAKE"
        confidence = prob_fake * 100
    else:
        prediction = "REAL"
        confidence = (1 - prob_fake) * 100
    
    # Generate Grad-CAM
    gradcam = get_gradcam(model)
    heatmap = gradcam.generate(input_tensor)
    
    # Create visualizations
    colored_heatmap = GradCAM.apply_colormap(heatmap)
    overlay = GradCAM.overlay_heatmap(original_image, heatmap, alpha=0.5)
    
    return overlay, colored_heatmap, confidence, prediction


# =============================================================================
# MODULE TEST
# =============================================================================

if __name__ == "__main__":
    """
    Test Grad-CAM functionality.
    
    Run: python gradcam.py
    """
    import os
    from model import get_model
    
    print("\n" + "="*60)
    print("TESTING GRAD-CAM MODULE")
    print("="*60)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️  Using device: {device}")
    
    # Check if model exists
    model_path = 'model.pth'
    if os.path.exists(model_path):
        # Load trained model
        print(f"\n📂 Loading trained model from: {model_path}")
        model = get_model(pretrained=False, freeze_backbone=False)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)
        print("✓ Model loaded successfully!")
    else:
        # Use untrained model for testing
        print("\n⚠️ No trained model found. Using pretrained model for testing.")
        model = get_model(pretrained=True, freeze_backbone=False)
        model = model.to(device)
    
    # Create test image (random noise for testing)
    print("\n🖼️ Creating test image...")
    test_image = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
    
    # Generate Grad-CAM
    print("\n🔥 Generating Grad-CAM visualization...")
    try:
        overlay, heatmap, confidence, prediction = generate_gradcam_visualization(
            model, test_image, device
        )
        
        print(f"\n📊 Results:")
        print(f"   - Prediction: {prediction}")
        print(f"   - Confidence: {confidence:.2f}%")
        print(f"   - Overlay shape: {overlay.shape}")
        print(f"   - Heatmap shape: {heatmap.shape}")
        
        # Save test outputs
        Image.fromarray(overlay).save('test_gradcam_overlay.png')
        Image.fromarray(heatmap).save('test_gradcam_heatmap.png')
        
        print(f"\n✓ Test images saved:")
        print(f"   - test_gradcam_overlay.png")
        print(f"   - test_gradcam_heatmap.png")
        
        print("\n✅ Grad-CAM module test successful!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print()
