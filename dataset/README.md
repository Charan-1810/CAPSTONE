# Dataset Directory Structure

This directory should contain your deepfake detection dataset.

## Required Structure

```
dataset/
├── real/
│   ├── image001.jpg
│   ├── image002.png
│   └── ...
└── fake/
    ├── image001.jpg
    ├── image002.png
    └── ...
```

## Instructions

1. **Create subfolders:**
   - `real/` - Contains genuine, unmanipulated face images
   - `fake/` - Contains deepfake/AI-generated face images

2. **Supported formats:**
   - JPG/JPEG
   - PNG
   - BMP
   - WebP

3. **Labels:**
   - Images in `real/` folder → Label 0
   - Images in `fake/` folder → Label 1

## Dataset Recommendations

For good model performance, consider:

- **Minimum samples:** 1000+ images per class
- **Balance:** Roughly equal number of real and fake images
- **Diversity:** Include various face angles, lighting, ethnicities
- **Quality:** Use clear, well-lit face images

## Suggested Datasets

You can download public deepfake datasets:

1. **FaceForensics++** - https://github.com/ondyari/FaceForensics
2. **DFDC (Deepfake Detection Challenge)** - https://ai.meta.com/datasets/dfdc/
3. **Celeb-DF** - https://github.com/yuezunli/celeb-deepfakeforensics
4. **FF++ (Face2Face, FaceSwap, etc.)**

## Notes

- Place this README in the dataset folder for reference
- The training script automatically handles stratified splitting
- Images will be resized to 224x224 during preprocessing
