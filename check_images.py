"""Quick script to preview downloaded images."""
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Check pixel values
real_img = Image.open('dataset/real/real_0000.jpg')
print(f'Real image: {real_img.size}, mode: {real_img.mode}')
print(f'Pixel sample (center): {real_img.getpixel((112, 112))}')

fake_img = Image.open('dataset/fake/fake_0000.jpg')
print(f'Fake image: {fake_img.size}, mode: {fake_img.mode}')
print(f'Pixel sample (center): {fake_img.getpixel((112, 112))}')

# Create preview grid
fig, axes = plt.subplots(2, 4, figsize=(12, 6))
for i in range(4):
    real = Image.open(f'dataset/real/real_{i:04d}.jpg')
    axes[0, i].imshow(real)
    axes[0, i].set_title(f'Real {i}')
    axes[0, i].axis('off')
    
    fake = Image.open(f'dataset/fake/fake_{i:04d}.jpg')
    axes[1, i].imshow(fake)
    axes[1, i].set_title(f'Fake {i}')
    axes[1, i].axis('off')

plt.suptitle('Sample Images - Top: Real, Bottom: Fake')
plt.tight_layout()
plt.savefig('sample_preview.png', dpi=100)
print('Preview saved to sample_preview.png')
