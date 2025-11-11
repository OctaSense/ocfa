#!/usr/bin/env python3
"""
OCFA Face SDK - Public RGB-IR Dataset Downloader

Downloads public RGB-IR face datasets for development and testing.
Supports multiple datasets with automatic download and organization.
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path
import json
import hashlib


# Dataset configurations
DATASETS = {
    'uncc_thermal': {
        'name': 'UNCC ThermalFace Dataset',
        'description': '~10,000 thermal images, 10 subjects, with landmarks',
        'url': 'https://drive.google.com/drive/folders/1UE4BsRsJBSEyLtcAkMQ_jYIBKwKjwyQc',
        'size': '~5-10 GB',
        'type': 'google_drive',
        'folder_id': '1UE4BsRsJBSEyLtcAkMQ_jYIBKwKjwyQc',
        'output_dir': 'uncc_thermal',
        'modality': 'thermal',
        'homepage': 'https://github.com/TeCSAR-UNCC/UNCC-ThermalFace'
    },
    'tfw': {
        'name': 'TFW: Thermal Faces in the Wild',
        'description': 'Annotated thermal face dataset',
        'url': 'https://github.com/IS2AI/TFW',
        'size': 'Variable',
        'type': 'github',
        'output_dir': 'tfw',
        'modality': 'thermal',
        'homepage': 'https://github.com/IS2AI/TFW'
    },
    'otcbvs_samples': {
        'name': 'OTCBVS Benchmark Samples',
        'description': 'Sample RGB-Thermal pairs from OTCBVS',
        'url': 'http://vcipl-okstate.org/pbvs/bench/Data/04/download.html',
        'size': '~500 MB',
        'type': 'manual',
        'output_dir': 'otcbvs',
        'modality': 'rgb-thermal',
        'homepage': 'https://vcipl-okstate.org/pbvs/bench/'
    },
    'lfw': {
        'name': 'Labeled Faces in the Wild (RGB only)',
        'description': 'Standard face recognition benchmark',
        'url': 'http://vis-www.cs.umass.edu/lfw/lfw.tgz',
        'size': '~180 MB',
        'type': 'direct',
        'output_dir': 'lfw',
        'modality': 'rgb',
        'homepage': 'http://vis-www.cs.umass.edu/lfw/'
    }
}


def check_gdown():
    """Check if gdown is installed"""
    try:
        subprocess.run(['gdown', '--version'],
                      capture_output=True,
                      check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def install_gdown():
    """Install gdown for Google Drive downloads"""
    print("\n  Installing gdown for Google Drive downloads...")
    try:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'gdown'],
                      check=True)
        print("  ✓ gdown installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("  ✗ Failed to install gdown")
        return False


def download_google_drive(folder_id, output_dir, folder_name):
    """Download from Google Drive using gdown"""
    print(f"\n  Downloading from Google Drive...")
    print(f"  Folder ID: {folder_id}")
    print(f"  Output: {output_dir}")

    # Check gdown
    if not check_gdown():
        print("\n  gdown is not installed.")
        if not install_gdown():
            return False

    try:
        # Download folder
        cmd = [
            'gdown',
            '--folder',
            f'https://drive.google.com/drive/folders/{folder_id}',
            '-O', str(output_dir),
            '--remaining-ok'
        ]

        print(f"\n  Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=True, capture_output=False)

        print(f"\n  ✓ Download completed")
        return True

    except subprocess.CalledProcessError as e:
        print(f"\n  ✗ Download failed: {e}")
        return False


def download_direct(url, output_dir):
    """Download directly using wget or curl"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = url.split('/')[-1]
    output_file = output_dir / filename

    print(f"\n  Downloading from: {url}")
    print(f"  Saving to: {output_file}")

    # Try wget first
    try:
        subprocess.run(['wget', '-O', str(output_file), url],
                      check=True)
        print(f"  ✓ Downloaded successfully")

        # Extract if tar/zip
        if filename.endswith(('.tgz', '.tar.gz', '.zip')):
            print(f"\n  Extracting {filename}...")
            if filename.endswith('.zip'):
                subprocess.run(['unzip', '-q', str(output_file), '-d', str(output_dir)],
                             check=True)
            else:
                subprocess.run(['tar', '-xzf', str(output_file), '-C', str(output_dir)],
                             check=True)
            print(f"  ✓ Extracted successfully")

        return True

    except (subprocess.CalledProcessError, FileNotFoundError):
        # Try curl
        try:
            subprocess.run(['curl', '-L', '-o', str(output_file), url],
                          check=True)
            print(f"  ✓ Downloaded successfully")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            print(f"  ✗ Download failed (need wget or curl)")
            return False


def clone_github_repo(url, output_dir):
    """Clone GitHub repository"""
    output_dir = Path(output_dir)

    print(f"\n  Cloning from GitHub: {url}")
    print(f"  Output: {output_dir}")

    try:
        subprocess.run(['git', 'clone', url, str(output_dir)],
                      check=True)
        print(f"  ✓ Cloned successfully")
        return True
    except subprocess.CalledProcessError:
        print(f"  ✗ Clone failed")
        return False


def download_dataset(dataset_key, base_dir):
    """Download a specific dataset"""
    if dataset_key not in DATASETS:
        print(f"✗ Unknown dataset: {dataset_key}")
        return False

    config = DATASETS[dataset_key]
    output_dir = Path(base_dir) / config['output_dir']

    print(f"\n{'='*70}")
    print(f"Dataset: {config['name']}")
    print(f"{'='*70}")
    print(f"Description: {config['description']}")
    print(f"Size: {config['size']}")
    print(f"Modality: {config['modality']}")
    print(f"Homepage: {config['homepage']}")

    # Check if already exists
    if output_dir.exists() and any(output_dir.iterdir()):
        print(f"\n  ⚠ Dataset already exists at: {output_dir}")
        response = input("  Overwrite? [y/N]: ").strip().lower()
        if response != 'y':
            print("  Skipping download")
            return True

    output_dir.mkdir(parents=True, exist_ok=True)

    # Download based on type
    download_type = config['type']

    if download_type == 'google_drive':
        return download_google_drive(
            config['folder_id'],
            output_dir,
            config['name']
        )

    elif download_type == 'direct':
        return download_direct(config['url'], output_dir)

    elif download_type == 'github':
        return clone_github_repo(config['url'], output_dir)

    elif download_type == 'manual':
        print(f"\n  ⚠ This dataset requires manual download")
        print(f"  Please visit: {config['homepage']}")
        print(f"  Download URL: {config['url']}")
        print(f"  Save to: {output_dir}")
        return False

    else:
        print(f"  ✗ Unknown download type: {download_type}")
        return False


def list_datasets():
    """List all available datasets"""
    print("\n" + "="*70)
    print("Available RGB-IR Face Datasets")
    print("="*70)

    for key, config in DATASETS.items():
        print(f"\n{key}:")
        print(f"  Name: {config['name']}")
        print(f"  Description: {config['description']}")
        print(f"  Size: {config['size']}")
        print(f"  Modality: {config['modality']}")
        print(f"  Type: {config['type']}")
        print(f"  Homepage: {config['homepage']}")

    print("\n" + "="*70)


def create_dataset_info(base_dir):
    """Create dataset information file"""
    info_file = Path(base_dir) / 'datasets_info.json'

    info = {
        'datasets': DATASETS,
        'downloaded': [],
        'base_dir': str(base_dir)
    }

    # Check which datasets exist
    for key, config in DATASETS.items():
        dataset_dir = Path(base_dir) / config['output_dir']
        if dataset_dir.exists() and any(dataset_dir.iterdir()):
            info['downloaded'].append(key)

    with open(info_file, 'w') as f:
        json.dump(info, f, indent=2)

    print(f"\n✓ Dataset info saved to: {info_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Download public RGB-IR face datasets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all available datasets
  python download_public_datasets.py --list

  # Download UNCC ThermalFace dataset
  python download_public_datasets.py --dataset uncc_thermal

  # Download multiple datasets
  python download_public_datasets.py --dataset uncc_thermal lfw

  # Download to custom directory
  python download_public_datasets.py --dataset uncc_thermal --output /path/to/datasets
        """
    )

    parser.add_argument('--list', action='store_true',
                       help='List all available datasets')
    parser.add_argument('--dataset', nargs='+', choices=list(DATASETS.keys()),
                       help='Dataset(s) to download')
    parser.add_argument('--output', default='datasets',
                       help='Output directory (default: ./datasets)')

    args = parser.parse_args()

    # List datasets
    if args.list:
        list_datasets()
        return 0

    # Check if dataset specified
    if not args.dataset:
        print("Error: Please specify --dataset or --list")
        parser.print_help()
        return 1

    # Create output directory
    base_dir = Path(args.output)
    base_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print("OCFA Face SDK - Public Dataset Downloader")
    print(f"{'='*70}")
    print(f"Output directory: {base_dir.absolute()}")
    print(f"Datasets to download: {len(args.dataset)}")

    # Download each dataset
    success_count = 0
    failed_datasets = []

    for dataset_key in args.dataset:
        success = download_dataset(dataset_key, base_dir)
        if success:
            success_count += 1
        else:
            failed_datasets.append(dataset_key)

    # Create dataset info
    create_dataset_info(base_dir)

    # Summary
    print(f"\n{'='*70}")
    print("Download Summary")
    print(f"{'='*70}")
    print(f"Total: {len(args.dataset)}")
    print(f"Success: {success_count}")
    print(f"Failed: {len(failed_datasets)}")

    if failed_datasets:
        print(f"\nFailed datasets: {', '.join(failed_datasets)}")

    if success_count > 0:
        print(f"\n✓ Successfully downloaded {success_count} dataset(s)!")
        print(f"\nDatasets location: {base_dir.absolute()}")
        print(f"\nNext steps:")
        print(f"  1. Check datasets_info.json for details")
        print(f"  2. Use the data for training/testing")
        print(f"  3. See data/samples/README.md for usage examples")

    return 0 if len(failed_datasets) == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
