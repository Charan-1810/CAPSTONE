"""
=============================================================================
DOWNLOAD_SAMPLE_DATA.PY - Download Sample Images for Testing
=============================================================================

This script downloads sample face images for testing the deepfake detection
pipeline. It uses the Labeled Faces in the Wild (LFW) dataset for real faces
and generates synthetic "fake" images by applying transformations.

For a real deepfake detection system, you would use actual deepfake datasets
like FaceForensics++, DFDC, or Celeb-DF.

NOTE: This is for TESTING/DEMO purposes only. The "fake" images generated
here are NOT real deepfakes - they are simply transformed versions of real
images to test the pipeline functionality.
"""

import os
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
from pathlib import Path
import urllib.request
import tarfile
import shutil
from tqdm import tqdm


def download_lfw_sample(data_dir: str, num_images: int = 100):
    """
    Download sample images from Labeled Faces in the Wild dataset.
    
    Args:
        data_dir: Directory to save images
        num_images: Number of images to download for each class
    """
    print("\n" + "="*60)
    print("DOWNLOADING SAMPLE DATA")
    print("="*60)
    
    real_dir = Path(data_dir) / 'real'
    fake_dir = Path(data_dir) / 'fake'
    
    # Create directories
    real_dir.mkdir(parents=True, exist_ok=True)
    fake_dir.mkdir(parents=True, exist_ok=True)
    
    # Clear existing images
    for f in real_dir.glob('*.jpg'):
        f.unlink()
    for f in fake_dir.glob('*.jpg'):
        f.unlink()
    
    # Try to use sklearn's LFW dataset
    try:
        from sklearn.datasets import fetch_lfw_people
        
        print("\n📥 Downloading LFW dataset (this may take a few minutes)...")
        print("   Source: Labeled Faces in the Wild")
        
        # Fetch LFW dataset with color images
        lfw_people = fetch_lfw_people(
            min_faces_per_person=20,
            resize=1.0,
            color=True,
            download_if_missing=True
        )
        
        images = lfw_people.images
        n_samples = min(num_images, len(images))
        
        print(f"\n✓ Downloaded {len(images)} images")
        print(f"   Image shape: {images[0].shape}")
        print(f"   Image dtype: {images[0].dtype}")
        print(f"   Image range: [{images[0].min():.2f}, {images[0].max():.2f}]")
        print(f"   Using {n_samples} images for real class")
        print(f"   Generating {n_samples} synthetic 'fake' images")
        
        # Save real images
        print("\n📁 Saving real images...")
        for i in tqdm(range(n_samples), desc="Real images"):
            img_array = images[i]
            
            # Normalize to 0-255 range if needed
            if img_array.max() <= 1.0:
                img_array = (img_array * 255).astype(np.uint8)
            else:
                img_array = img_array.astype(np.uint8)
            
            img = Image.fromarray(img_array, mode='RGB')
            img = img.resize((224, 224), Image.Resampling.LANCZOS)
            img.save(real_dir / f'real_{i:04d}.jpg', 'JPEG', quality=95)
        
        # Generate synthetic "fake" images (transformed versions)
        print("\n📁 Generating synthetic 'fake' images...")
        print("   (Applying various transformations to simulate artifacts)")
        
        for i in tqdm(range(n_samples), desc="Fake images"):
            img_array = images[(i + n_samples // 2) % len(images)]
            
            # Normalize to 0-255 range if needed
            if img_array.max() <= 1.0:
                img_array = (img_array * 255).astype(np.uint8)
            else:
                img_array = img_array.astype(np.uint8)
            
            img = Image.fromarray(img_array, mode='RGB')
            img = img.resize((224, 224), Image.Resampling.LANCZOS)
            
            # Apply transformations to simulate "fake" artifacts
            img = apply_fake_artifacts(img, i)
            img.save(fake_dir / f'fake_{i:04d}.jpg', 'JPEG', quality=85)
        
        print_summary(real_dir, fake_dir)
        return True
        
    except ImportError:
        print("❌ sklearn not found. Trying alternative method...")
        return download_alternative(data_dir, num_images)
    except Exception as e:
        print(f"❌ LFW download failed: {e}")
        import traceback
        traceback.print_exc()
        print("   Trying alternative method...")
        return download_alternative(data_dir, num_images)


def apply_fake_artifacts(img: Image.Image, seed: int) -> Image.Image:
    """
    Apply various transformations to simulate deepfake artifacts.
    
    Real deepfakes have specific artifacts, but for testing purposes,
    we apply transformations that create visual differences the model
    can learn to detect.
    
    Args:
        img: PIL Image
        seed: Random seed for reproducibility
    
    Returns:
        Transformed PIL Image
    """
    np.random.seed(seed)
    
    # Apply MULTIPLE artifacts to make fakes more distinguishable
    img_array = np.array(img).astype(np.float32)
    
    # 1. Always add some blur (common in deepfakes)
    img = img.filter(ImageFilter.GaussianBlur(radius=1.2))
    
    # 2. Color shift (simulates color inconsistencies)
    enhancer = ImageEnhance.Color(img)
    img = enhancer.enhance(np.random.uniform(1.15, 1.35))
    
    # 3. Slight contrast adjustment
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(np.random.uniform(1.1, 1.25))
    
    # 4. Add compression artifacts
    from io import BytesIO
    buffer = BytesIO()
    img.save(buffer, format='JPEG', quality=np.random.randint(40, 60))
    buffer.seek(0)
    img = Image.open(buffer).copy()
    
    # 5. Add subtle noise
    img_array = np.array(img).astype(np.float32)
    noise = np.random.normal(0, np.random.uniform(5, 12), img_array.shape)
    img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(img_array)
    
    # 6. Additional artifact based on seed
    artifact_type = seed % 3
    if artifact_type == 0:
        # Extra sharpening (creates unnatural edges)
        img = img.filter(ImageFilter.SHARPEN)
    elif artifact_type == 1:
        # Slight brightness shift
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(np.random.uniform(0.85, 0.95))
    else:
        # Edge enhancement (common artifact)
        img = img.filter(ImageFilter.EDGE_ENHANCE)
    
    return img


def download_alternative(data_dir: str, num_images: int = 100):
    """
    Alternative method: Generate synthetic face-like images for testing.
    
    This creates simple geometric patterns that resemble faces.
    For real training, use actual face datasets.
    """
    print("\n📥 Generating synthetic test images...")
    
    real_dir = Path(data_dir) / 'real'
    fake_dir = Path(data_dir) / 'fake'
    
    real_dir.mkdir(parents=True, exist_ok=True)
    fake_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n   Generating {num_images} 'real' images...")
    print(f"   Generating {num_images} 'fake' images...")
    
    for i in tqdm(range(num_images), desc="Generating images"):
        # Create "real" image (smoother gradients)
        real_img = generate_face_pattern(i, is_fake=False)
        real_img.save(real_dir / f'real_{i:04d}.jpg', 'JPEG', quality=95)
        
        # Create "fake" image (with artifacts)
        fake_img = generate_face_pattern(i + 1000, is_fake=True)
        fake_img.save(fake_dir / f'fake_{i:04d}.jpg', 'JPEG', quality=85)
    
    print_summary(real_dir, fake_dir)
    return True


def generate_face_pattern(seed: int, is_fake: bool = False) -> Image.Image:
    """
    Generate a synthetic face-like pattern for testing.
    
    Args:
        seed: Random seed
        is_fake: If True, add artifacts
    
    Returns:
        PIL Image (224x224)
    """
    np.random.seed(seed)
    
    # Create base face-like pattern
    size = 224
    img = np.zeros((size, size, 3), dtype=np.float32)
    
    # Background color
    bg_color = np.random.uniform(0.6, 0.9, 3)
    img[:] = bg_color * 255
    
    # Face oval
    center_x, center_y = size // 2, size // 2
    y, x = np.ogrid[:size, :size]
    
    # Ellipse for face
    face_mask = ((x - center_x) ** 2 / (70 ** 2) + (y - center_y - 10) ** 2 / (90 ** 2)) <= 1
    skin_color = np.array([np.random.uniform(180, 230), 
                          np.random.uniform(150, 200), 
                          np.random.uniform(120, 170)])
    img[face_mask] = skin_color
    
    # Eyes (simple dark circles)
    for eye_x in [center_x - 25, center_x + 25]:
        eye_y = center_y - 15
        eye_mask = ((x - eye_x) ** 2 + (y - eye_y) ** 2) <= 100
        img[eye_mask] = [50, 50, 50]
    
    # Mouth (simple dark line area)
    mouth_mask = (abs(y - (center_y + 35)) < 5) & (abs(x - center_x) < 25)
    img[mouth_mask] = [120, 80, 80]
    
    if is_fake:
        # Add artifacts for "fake" images
        # Slight blur
        from scipy.ndimage import gaussian_filter
        img = gaussian_filter(img, sigma=1.5)
        
        # Add noise
        noise = np.random.normal(0, 8, img.shape)
        img = np.clip(img + noise, 0, 255)
        
        # Color shift
        img[:, :, 0] = np.clip(img[:, :, 0] * 1.1, 0, 255)
    else:
        # Smooth real images
        from scipy.ndimage import gaussian_filter
        img = gaussian_filter(img, sigma=0.5)
    
    img = np.clip(img, 0, 255).astype(np.uint8)
    return Image.fromarray(img)


def print_summary(real_dir: Path, fake_dir: Path):
    """Print summary of downloaded/generated data."""
    real_count = len(list(real_dir.glob('*.jpg')))
    fake_count = len(list(fake_dir.glob('*.jpg')))
    
    print("\n" + "="*60)
    print("✅ SAMPLE DATA READY!")
    print("="*60)
    print(f"\n📊 Dataset Summary:")
    print(f"   - Real images: {real_count} (in dataset/real/)")
    print(f"   - Fake images: {fake_count} (in dataset/fake/)")
    print(f"   - Total: {real_count + fake_count} images")
    
    print("\n📋 Next Steps:")
    print("   1. Run: python train.py")
    print("   2. Run: python evaluate.py")
    print("   3. Run: streamlit run app.py")
    
    print("\n⚠️  NOTE: This is synthetic/transformed data for TESTING only.")
    print("   For production, use real deepfake datasets like:")
    print("   - FaceForensics++ (https://github.com/ondyari/FaceForensics)")
    print("   - DFDC (https://ai.meta.com/datasets/dfdc/)")
    print("   - Celeb-DF (https://github.com/yuezunli/celeb-deepfakeforensics)")
    print()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Download sample data for deepfake detection')
    parser.add_argument('--num-images', type=int, default=100,
                        help='Number of images per class (default: 100)')
    parser.add_argument('--data-dir', type=str, default='dataset',
                        help='Dataset directory (default: dataset)')
    
    args = parser.parse_args()
    
    download_lfw_sample(args.data_dir, args.num_images)
