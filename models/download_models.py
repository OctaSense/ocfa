"""
OCFA Face SDK - Model Download Script

Download pre-trained models from InsightFace.
"""

import os
import sys
import urllib.request
from pathlib import Path


MODELS = {
    'arcface_r100': {
        'url': 'https://github.com/onnx/models/raw/main/validated/vision/body_analysis/arcface/model/arcfaceresnet100-8.onnx',
        'filename': 'arcface_r100.onnx',
        'size_mb': 249,
        'description': 'ArcFace-R100 face recognition model (ONNX)'
    },
    'arcface_mobilefacenet': {
        'url': 'https://github.com/onnx/models/raw/main/validated/vision/body_analysis/arcface/model/arcfacemobilefacenet-11.onnx',
        'filename': 'arcface_mobilefacenet.onnx',
        'size_mb': 3.8,
        'description': 'ArcFace MobileFaceNet (lightweight, ONNX)'
    },
    # Note: MiniFASNet ONNX model needs to be trained or converted from PyTorch
    # Alternative: Use pre-trained model from Silent-Face-Anti-Spoofing
}


def download_file(url: str, output_path: str, description: str):
    """
    Download file with progress bar

    Args:
        url: Download URL
        output_path: Output file path
        description: File description
    """
    print(f"\nDownloading {description}...")
    print(f"URL: {url}")
    print(f"Output: {output_path}")

    def reporthook(count, block_size, total_size):
        """Progress callback"""
        percent = int(count * block_size * 100 / total_size)
        sys.stdout.write(f"\r[{'=' * (percent // 2)}{' ' * (50 - percent // 2)}] {percent}%")
        sys.stdout.flush()

    try:
        urllib.request.urlretrieve(url, output_path, reporthook)
        print("\n✓ Download complete")
        return True
    except Exception as e:
        print(f"\n✗ Download failed: {e}")
        return False


def main():
    """Main download function"""
    script_dir = Path(__file__).parent
    os.chdir(script_dir)

    print("=" * 60)
    print("OCFA Face SDK - Model Downloader")
    print("=" * 60)

    # Check if models directory exists
    if not os.path.exists('.'):
        os.makedirs('.')

    # Download all models
    for model_name, model_info in MODELS.items():
        output_path = model_info['filename']

        # Check if file already exists
        if os.path.exists(output_path):
            file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
            print(f"\n{model_name}: Already exists ({file_size_mb:.1f} MB)")
            continue

        # Download
        success = download_file(
            model_info['url'],
            output_path,
            f"{model_name} ({model_info['description']})"
        )

        if not success:
            print(f"Failed to download {model_name}")
            continue

    print("\n" + "=" * 60)
    print("Download Summary:")
    print("=" * 60)

    for model_name, model_info in MODELS.items():
        output_path = model_info['filename']
        if os.path.exists(output_path):
            file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
            print(f"✓ {model_name}: {output_path} ({file_size_mb:.1f} MB)")
        else:
            print(f"✗ {model_name}: Not downloaded")

    print("\n" + "=" * 60)
    print("Next Steps:")
    print("1. Export models to ONNX:")
    print("   cd ../python/tools")
    print("   python model_export.py")
    print("\n2. Quantize models to INT8:")
    print("   python quantization.py")
    print("=" * 60)


if __name__ == '__main__':
    main()
