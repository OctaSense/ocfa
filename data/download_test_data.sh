#!/bin/bash
#
# OCFA Face SDK - Test Data Download Script
#
# This script helps download RGB-IR face test datasets for development and testing.
#

set -e

echo "======================================================================"
echo "OCFA Face SDK - Test Data Downloader"
echo "======================================================================"
echo ""

DATA_DIR="$(dirname "$0")"
SAMPLES_DIR="$DATA_DIR/samples"

mkdir -p "$SAMPLES_DIR/rgb"
mkdir -p "$SAMPLES_DIR/ir"

echo "Available Public RGB-IR Face Datasets:"
echo ""
echo "1. OTCBVS Benchmark Dataset"
echo "   - 4,228 RGB-Thermal pairs from 30 subjects"
echo "   - URL: https://vcipl-okstate.org/pbvs/bench/"
echo "   - Download: Manual (registration required)"
echo ""
echo "2. Tufts Face Database (Kaggle)"
echo "   - 112 participants, ~100K images"
echo "   - URL: https://www.kaggle.com/datasets/kpvisionlab/tufts-face-database"
echo "   - Download: Requires Kaggle account"
echo "   - Command: kaggle datasets download -d kpvisionlab/tufts-face-database"
echo ""
echo "3. SYSU-MM01 Dataset"
echo "   - 22K visible + 11K infrared images, 491 identities"
echo "   - URL: https://github.com/wuancong/SYSU-MM01"
echo "   - Download: Contact authors for access"
echo ""
echo "4. Carl Dataset"
echo "   - 45,900 images (RGB + Depth + Thermal), 51 subjects"
echo "   - URL: https://github.com/TimSimpson/carl_dataset"
echo "   - Download: Check GitHub repository"
echo ""

echo "======================================================================"
echo "Quick Start: Download Sample Data"
echo "======================================================================"
echo ""

# Check if wget or curl is available
if command -v wget &> /dev/null; then
    DOWNLOADER="wget -O"
elif command -v curl &> /dev/null; then
    DOWNLOADER="curl -L -o"
else
    echo "Error: Neither wget nor curl is available. Please install one of them."
    exit 1
fi

echo "Downloading sample face images for testing..."
echo ""

# Download sample RGB images from public sources
echo "[1/4] Downloading RGB sample 1..."
$DOWNLOADER "$SAMPLES_DIR/rgb/sample_001.jpg" \
    "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg" \
    2>/dev/null || echo "  Failed to download sample 1"

echo "[2/4] Downloading RGB sample 2..."
$DOWNLOADER "$SAMPLES_DIR/rgb/sample_002.jpg" \
    "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/messi5.jpg" \
    2>/dev/null || echo "  Failed to download sample 2"

# Create synthetic IR samples (grayscale versions)
echo "[3/4] Creating synthetic IR samples..."
if command -v convert &> /dev/null; then
    for rgb_file in "$SAMPLES_DIR/rgb"/*.jpg; do
        if [ -f "$rgb_file" ]; then
            filename=$(basename "$rgb_file")
            convert "$rgb_file" -colorspace Gray "$SAMPLES_DIR/ir/${filename%.jpg}_ir.jpg" 2>/dev/null
        fi
    done
    echo "  ✓ Created IR samples"
else
    echo "  ⚠ ImageMagick not installed, skipping IR conversion"
    echo "    Install: sudo apt-get install imagemagick  (Linux)"
    echo "    Install: brew install imagemagick          (macOS)"
fi

echo "[4/4] Creating test data manifest..."
cat > "$DATA_DIR/samples/README.md" << 'EOF'
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
EOF

echo "  ✓ Created README.md"

echo ""
echo "======================================================================"
echo "✓ Setup Complete"
echo "======================================================================"
echo ""
echo "Downloaded samples to: $SAMPLES_DIR"
echo ""
echo "Next steps:"
echo "  1. Download full datasets from the URLs above (optional)"
echo "  2. Test the SDK with sample data:"
echo "     python python/examples/test_recognition.py"
echo ""
echo "For production use, please collect your own RGB-IR face dataset"
echo "that matches your target deployment environment."
echo ""
