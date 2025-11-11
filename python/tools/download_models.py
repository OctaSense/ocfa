#!/usr/bin/env python3
"""
OCFA Face SDK - Download Real InsightFace Models

Downloads pre-trained models from InsightFace project and official sources.
"""

import argparse
import sys
import os
from pathlib import Path
import urllib.request
import hashlib
import json

# Model download configurations
MODELS = {
    'buffalo_l': {
        'name': 'Buffalo_L (Complete Package)',
        'url': 'https://huggingface.co/public-data/insightface/resolve/main/models/buffalo_l.zip',
        'filename': 'buffalo_l.zip',
        'sha256': None,
        'size_mb': 289,
        'description': 'Complete face analysis suite: detection + ArcFace-R50 recognition + alignment + gender/age',
        'extract': True
    },
    'arcface_r100': {
        'name': 'ArcFace-R100 (Best Accuracy)',
        'url': 'https://huggingface.co/public-data/insightface/resolve/main/models/arcface_r100_v1/arcface_r100_v1.onnx',
        'filename': 'arcface_r100_v1.onnx',
        'sha256': None,
        'size_mb': 249,
        'description': 'ArcFace ResNet-100 face recognition model (highest accuracy)'
    },
    'retinaface_r50': {
        'name': 'RetinaFace-R50',
        'url': 'https://huggingface.co/public-data/insightface/resolve/main/models/retinaface_r50_v1/retinaface_r50_v1.onnx',
        'filename': 'retinaface_r50_v1.onnx',
        'sha256': None,
        'size_mb': 105,
        'description': 'RetinaFace ResNet-50 face detector'
    },
    'scrfd_10g': {
        'name': 'SCRFD-10G (Face Detector)',
        'url': 'https://huggingface.co/public-data/insightface/resolve/main/models/scrfd_10g.onnx',
        'filename': 'scrfd_10g.onnx',
        'sha256': None,
        'size_mb': 17,
        'description': 'SCRFD face detector (10G variant, high accuracy)'
    },
    'scrfd_2.5g': {
        'name': 'SCRFD-2.5G (Face Detector)',
        'url': 'https://huggingface.co/public-data/insightface/resolve/main/models/scrfd_2.5g.onnx',
        'filename': 'scrfd_2.5g.onnx',
        'sha256': None,
        'size_mb': 3.7,
        'description': 'SCRFD face detector (2.5G variant, balanced)'
    },
    'scrfd_person_2.5g': {
        'name': 'SCRFD Person Detector',
        'url': 'https://huggingface.co/public-data/insightface/resolve/main/models/scrfd_person_2.5g.onnx',
        'filename': 'scrfd_person_2.5g.onnx',
        'sha256': None,
        'size_mb': 3.7,
        'description': 'SCRFD person detector (for full body detection)'
    }
}


def download_file(url, output_path, show_progress=True):
    """Download file with progress bar"""
    print(f"  Downloading from: {url}")
    print(f"  Saving to: {output_path}")

    def report_progress(block_num, block_size, total_size):
        if not show_progress:
            return
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(100, downloaded * 100 / total_size)
            downloaded_mb = downloaded / (1024 * 1024)
            total_mb = total_size / (1024 * 1024)
            print(f"\r  Progress: {percent:.1f}% ({downloaded_mb:.1f}/{total_mb:.1f} MB)", end='')

    try:
        urllib.request.urlretrieve(url, output_path, reporthook=report_progress)
        if show_progress:
            print()  # New line after progress
        print(f"  ✓ Download completed")
        return True
    except Exception as e:
        print(f"\n  ✗ Download failed: {e}")
        if os.path.exists(output_path):
            os.remove(output_path)
        return False


def verify_sha256(file_path, expected_hash):
    """Verify file SHA256 checksum"""
    if expected_hash is None:
        print(f"  ⚠ Skipping checksum verification (no hash provided)")
        return True

    print(f"  Verifying checksum...")
    sha256 = hashlib.sha256()

    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha256.update(chunk)

    actual_hash = sha256.hexdigest()
    if actual_hash == expected_hash:
        print(f"  ✓ Checksum verified")
        return True
    else:
        print(f"  ✗ Checksum mismatch!")
        print(f"    Expected: {expected_hash}")
        print(f"    Actual:   {actual_hash}")
        return False


def extract_archive(archive_path, output_dir):
    """Extract zip/tar archive"""
    import zipfile
    import tarfile

    print(f"  Extracting {archive_path}...")

    try:
        if archive_path.endswith('.zip'):
            with zipfile.ZipFile(archive_path, 'r') as zf:
                zf.extractall(output_dir)
        elif archive_path.endswith(('.tar.gz', '.tgz')):
            with tarfile.open(archive_path, 'r:gz') as tf:
                tf.extractall(output_dir)
        elif archive_path.endswith('.tar'):
            with tarfile.open(archive_path, 'r') as tf:
                tf.extractall(output_dir)

        print(f"  ✓ Extraction completed")
        return True
    except Exception as e:
        print(f"  ✗ Extraction failed: {e}")
        return False


def download_model(model_key, output_dir, force=False):
    """Download a specific model"""
    if model_key not in MODELS:
        print(f"✗ Unknown model: {model_key}")
        print(f"  Available models: {', '.join(MODELS.keys())}")
        return False

    model_info = MODELS[model_key]
    output_path = Path(output_dir) / model_info['filename']

    print(f"\n{'='*60}")
    print(f"Model: {model_info['name']}")
    print(f"Description: {model_info['description']}")
    print(f"Size: ~{model_info['size_mb']} MB")
    print(f"{'='*60}")

    # Check if already exists
    if output_path.exists() and not force:
        print(f"  ⚠ File already exists: {output_path}")
        response = input("  Overwrite? [y/N]: ").strip().lower()
        if response != 'y':
            print(f"  Skipping download")
            return True

    # Download
    success = download_file(model_info['url'], str(output_path))
    if not success:
        return False

    # Verify checksum
    if model_info['sha256']:
        if not verify_sha256(str(output_path), model_info['sha256']):
            os.remove(output_path)
            return False

    # Extract if needed
    if model_info.get('extract', False):
        extract_dir = output_path.parent / output_path.stem
        extract_dir.mkdir(exist_ok=True)
        if not extract_archive(str(output_path), str(extract_dir)):
            return False

    print(f"✓ Model '{model_key}' downloaded successfully\n")
    return True


def list_models():
    """List all available models"""
    print("\n" + "="*60)
    print("Available InsightFace Models")
    print("="*60)

    for key, info in MODELS.items():
        print(f"\n{key}:")
        print(f"  Name: {info['name']}")
        print(f"  Description: {info['description']}")
        print(f"  Size: ~{info['size_mb']} MB")
        print(f"  URL: {info['url']}")

    print("\n" + "="*60)


def main():
    parser = argparse.ArgumentParser(
        description='Download pre-trained InsightFace models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all available models
  python download_models.py --list

  # Download ArcFace-R34
  python download_models.py --model arcface_r34

  # Download multiple models
  python download_models.py --model arcface_r34 arcface_r50

  # Download all models
  python download_models.py --all

  # Force re-download
  python download_models.py --model arcface_r34 --force
        """
    )

    parser.add_argument('--list', action='store_true',
                        help='List all available models')
    parser.add_argument('--model', nargs='+', choices=list(MODELS.keys()),
                        help='Model(s) to download')
    parser.add_argument('--all', action='store_true',
                        help='Download all models')
    parser.add_argument('--output', default='../../models',
                        help='Output directory (default: ../../models)')
    parser.add_argument('--force', action='store_true',
                        help='Force re-download even if file exists')

    args = parser.parse_args()

    # List models
    if args.list:
        list_models()
        return 0

    # Determine which models to download
    models_to_download = []
    if args.all:
        models_to_download = list(MODELS.keys())
    elif args.model:
        models_to_download = args.model
    else:
        print("Error: Please specify --model, --all, or --list")
        parser.print_help()
        return 1

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"OCFA Face SDK - Model Downloader")
    print(f"{'='*60}")
    print(f"Output directory: {output_dir.absolute()}")
    print(f"Models to download: {len(models_to_download)}")

    # Download each model
    success_count = 0
    failed_models = []

    for model_key in models_to_download:
        success = download_model(model_key, output_dir, force=args.force)
        if success:
            success_count += 1
        else:
            failed_models.append(model_key)

    # Summary
    print(f"\n{'='*60}")
    print(f"Download Summary")
    print(f"{'='*60}")
    print(f"Total: {len(models_to_download)}")
    print(f"Success: {success_count}")
    print(f"Failed: {len(failed_models)}")

    if failed_models:
        print(f"\nFailed models: {', '.join(failed_models)}")
        return 1

    print(f"\n✓ All models downloaded successfully!")
    print(f"\nNext steps:")
    print(f"  1. Convert to ONNX if needed: python model_export.py")
    print(f"  2. Quantize to INT8: python quantization.py")
    print(f"  3. Validate models: python evaluate.py")

    return 0


if __name__ == '__main__':
    sys.exit(main())
