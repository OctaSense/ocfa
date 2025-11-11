# Test Data Samples

This directory contains sample RGB and IR face images for testing the OCFA Face SDK.

## Directory Structure

```
samples/
├── rgb/           # RGB face images
│   ├── sample_001.jpg
│   ├── sample_002.jpg
│   └── ...
└── ir/            # IR/thermal face images
    ├── sample_001_ir.jpg
    ├── sample_002_ir.jpg
    └── ...
```

## Data Sources

### Public Datasets (Recommended for Research)

1. **OTCBVS Benchmark Dataset**
   - URL: https://vcipl-okstate.org/pbvs/bench/
   - Size: 4,228 RGB-Thermal pairs, 30 subjects
   - License: Research use

2. **Tufts Face Database**
   - URL: https://www.kaggle.com/datasets/kpvisionlab/tufts-face-database
   - Size: ~100K images, 112 participants
   - License: Research use (Kaggle account required)

3. **SYSU-MM01**
   - URL: https://github.com/wuancong/SYSU-MM01
   - Size: 22K RGB + 11K NIR images, 491 identities
   - License: Contact authors

### Download Instructions

#### Option 1: Kaggle CLI (Tufts Dataset)

```bash
# Install Kaggle CLI
pip install kaggle

# Configure API credentials (create on kaggle.com)
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_api_key

# Download dataset
kaggle datasets download -d kpvisionlab/tufts-face-database
unzip tufts-face-database.zip -d samples/
```

#### Option 2: Manual Download

1. Visit the dataset URLs above
2. Register/login as required
3. Download the dataset files
4. Extract to `data/samples/`

## Usage in SDK

```python
from ocfa import OCFAFaceSDK
import cv2

sdk = OCFAFaceSDK(config_path='configs/default_config.json')

# Load RGB and IR images
rgb_image = cv2.imread('data/samples/rgb/sample_001.jpg')
ir_image = cv2.imread('data/samples/ir/sample_001_ir.jpg', cv2.IMREAD_GRAYSCALE)

# Run recognition
result = sdk.recognize(rgb_image, ir_image)

if result.success:
    print(f"Liveness: {result.liveness_score:.3f}")
    print(f"Quality: {result.quality_score:.3f}")
    print(f"Feature shape: {result.feature.shape}")
```

## Notes

- Sample images in this directory are for testing purposes only
- For production use, collect your own RGB-IR face dataset
- Ensure proper lighting and camera calibration for RGB-IR pairs
- Follow dataset licenses and terms of use

## Image Requirements

### RGB Images
- Format: JPEG, PNG
- Size: Minimum 112x112 pixels
- Color space: RGB
- Face should occupy 60-80% of image

### IR Images
- Format: JPEG, PNG (grayscale)
- Size: Same as RGB (112x112+)
- Wavelength: Near-infrared (850nm typical)
- Should be temporally synchronized with RGB

## Creating Your Own Dataset

```bash
# Capture RGB-IR pairs
# Resize to standard size
for img in rgb/*.jpg; do
    convert "$img" -resize 640x480^ -gravity center -extent 640x480 "rgb_resized/$(basename $img)"
done

# Generate calibration data
python python/tools/create_calibration_data.py \
    --dual-stream \
    --num-samples 100 \
    --output data/samples
```
