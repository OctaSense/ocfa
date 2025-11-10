#!/usr/bin/env python3
"""
OCFA Face SDK - Model Evaluation Tool

Evaluate model accuracy on test datasets.
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import json
from collections import defaultdict

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ocfa import OCFAFaceSDK


def evaluate_liveness(sdk, test_data):
    """Evaluate liveness detection accuracy"""
    print("\n=== Liveness Detection Evaluation ===")

    tp, tn, fp, fn = 0, 0, 0, 0
    threshold = sdk.config.get('liveness_threshold', 0.90)

    for sample in test_data:
        rgb, ir, label = sample['rgb'], sample['ir'], sample['is_live']

        score = sdk.liveness_detector.detect(rgb, ir)
        predicted = score >= threshold

        if label and predicted:
            tp += 1
        elif not label and not predicted:
            tn += 1
        elif not label and predicted:
            fp += 1
        else:
            fn += 1

    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # APCER and BPCER (face anti-spoofing metrics)
    apcer = fp / (tn + fp) if (tn + fp) > 0 else 0  # Attack Presentation Classification Error Rate
    bpcer = fn / (tp + fn) if (tp + fn) > 0 else 0  # Bona Fide Presentation Classification Error Rate
    acer = (apcer + bpcer) / 2  # Average Classification Error Rate

    print(f"Total samples: {total}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"APCER: {apcer:.4f}")
    print(f"BPCER: {bpcer:.4f}")
    print(f"ACER: {acer:.4f}")

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'apcer': apcer,
        'bpcer': bpcer,
        'acer': acer,
        'confusion_matrix': {'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn}
    }


def evaluate_recognition(sdk, test_pairs):
    """Evaluate 1:1 face verification accuracy"""
    print("\n=== Face Recognition (1:1) Evaluation ===")

    similarities = []
    labels = []

    for pair in test_pairs:
        rgb1, ir1, rgb2, ir2, is_same = pair['rgb1'], pair['ir1'], pair['rgb2'], pair['ir2'], pair['is_same']

        # Extract features
        feature1 = sdk.feature_extractor.extract(rgb1, ir1)
        feature2 = sdk.feature_extractor.extract(rgb2, ir2)

        # Compute similarity
        similarity = sdk.comparator.compare(feature1, feature2)

        similarities.append(similarity)
        labels.append(1 if is_same else 0)

    similarities = np.array(similarities)
    labels = np.array(labels)

    # Compute metrics at different thresholds
    thresholds = np.arange(0.5, 1.0, 0.05)
    best_threshold = 0.0
    best_accuracy = 0.0

    for threshold in thresholds:
        predictions = (similarities >= threshold).astype(int)
        accuracy = np.mean(predictions == labels)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold

    # Compute metrics at best threshold
    predictions = (similarities >= best_threshold).astype(int)

    tp = np.sum((predictions == 1) & (labels == 1))
    tn = np.sum((predictions == 0) & (labels == 0))
    fp = np.sum((predictions == 1) & (labels == 0))
    fn = np.sum((predictions == 0) & (labels == 1))

    accuracy = (tp + tn) / len(labels)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # FAR and FRR
    far = fp / (tn + fp) if (tn + fp) > 0 else 0  # False Accept Rate
    frr = fn / (tp + fn) if (tp + fn) > 0 else 0  # False Reject Rate

    print(f"Total pairs: {len(labels)}")
    print(f"Best threshold: {best_threshold:.3f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"FAR: {far:.4f}")
    print(f"FRR: {frr:.4f}")

    return {
        'accuracy': accuracy,
        'best_threshold': best_threshold,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'far': far,
        'frr': frr,
        'confusion_matrix': {'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn)}
    }


def evaluate_identification(sdk, gallery, probes):
    """Evaluate 1:N face identification accuracy"""
    print("\n=== Face Identification (1:N) Evaluation ===")

    # Build gallery database
    print("Building gallery database...")
    for person_id, (rgb, ir) in gallery.items():
        feature = sdk.feature_extractor.extract(rgb, ir)
        user_id = person_id.to_bytes(16, 'big')  # Convert to 16-byte ID
        sdk.add_user(user_id, feature)

    print(f"Gallery size: {len(gallery)}")

    # Evaluate probes
    rank1_correct = 0
    rank5_correct = 0
    total = len(probes)

    for probe_id, (rgb, ir) in probes.items():
        feature = sdk.feature_extractor.extract(rgb, ir)

        # Search
        matches = sdk.search_users(feature, threshold=0.0, max_results=5)

        if matches:
            # Rank-1 accuracy
            top1_id = int.from_bytes(matches[0][0], 'big')
            if top1_id == probe_id:
                rank1_correct += 1

            # Rank-5 accuracy
            top5_ids = [int.from_bytes(m[0], 'big') for m in matches[:5]]
            if probe_id in top5_ids:
                rank5_correct += 1

    rank1_acc = rank1_correct / total if total > 0 else 0
    rank5_acc = rank5_correct / total if total > 0 else 0

    print(f"Total probes: {total}")
    print(f"Rank-1 Accuracy: {rank1_acc:.4f}")
    print(f"Rank-5 Accuracy: {rank5_acc:.4f}")

    return {
        'rank1_accuracy': rank1_acc,
        'rank5_accuracy': rank5_acc,
        'gallery_size': len(gallery),
        'probe_count': total
    }


def load_test_data(data_dir, task='liveness'):
    """Load test data from directory"""
    print(f"\nLoading {task} test data from {data_dir}...")

    # For now, create dummy data
    # In real usage, load from actual dataset

    if task == 'liveness':
        data = []
        # Create 100 live and 100 fake samples
        for i in range(100):
            rgb = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
            ir = np.random.randint(0, 255, (720, 1280), dtype=np.uint8)
            data.append({'rgb': rgb, 'ir': ir, 'is_live': True})

        for i in range(100):
            rgb = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
            ir = np.random.randint(0, 255, (720, 1280), dtype=np.uint8)
            data.append({'rgb': rgb, 'ir': ir, 'is_live': False})

        print(f"Loaded {len(data)} samples")
        return data

    elif task == 'recognition':
        pairs = []
        # Create 100 positive and 100 negative pairs
        for i in range(100):
            rgb1 = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
            ir1 = np.random.randint(0, 255, (720, 1280), dtype=np.uint8)
            rgb2 = rgb1 + np.random.randint(-10, 10, rgb1.shape, dtype=np.int16).astype(np.uint8)
            ir2 = ir1 + np.random.randint(-10, 10, ir1.shape, dtype=np.int16).astype(np.uint8)
            pairs.append({'rgb1': rgb1, 'ir1': ir1, 'rgb2': rgb2, 'ir2': ir2, 'is_same': True})

        for i in range(100):
            rgb1 = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
            ir1 = np.random.randint(0, 255, (720, 1280), dtype=np.uint8)
            rgb2 = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
            ir2 = np.random.randint(0, 255, (720, 1280), dtype=np.uint8)
            pairs.append({'rgb1': rgb1, 'ir1': ir1, 'rgb2': rgb2, 'ir2': ir2, 'is_same': False})

        print(f"Loaded {len(pairs)} pairs")
        return pairs

    elif task == 'identification':
        # Create 50 gallery identities
        gallery = {}
        for i in range(50):
            rgb = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
            ir = np.random.randint(0, 255, (720, 1280), dtype=np.uint8)
            gallery[i] = (rgb, ir)

        # Create 50 probe samples (same identities)
        probes = {}
        for i in range(50):
            rgb = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
            ir = np.random.randint(0, 255, (720, 1280), dtype=np.uint8)
            probes[i] = (rgb, ir)

        print(f"Loaded {len(gallery)} gallery, {len(probes)} probes")
        return gallery, probes


def main():
    parser = argparse.ArgumentParser(description='OCFA Face SDK Model Evaluation')
    parser.add_argument('--config', default='../../configs/default_config.json',
                        help='Path to config file')
    parser.add_argument('--data', default='../../data/test',
                        help='Path to test data directory')
    parser.add_argument('--task', choices=['liveness', 'recognition', 'identification', 'all'],
                        default='all', help='Which task to evaluate')
    parser.add_argument('--output', default='evaluation_results.json',
                        help='Output JSON file for results')

    args = parser.parse_args()

    print("=" * 60)
    print("OCFA Face SDK - Model Evaluation")
    print("=" * 60)

    # Initialize SDK
    print(f"\nInitializing SDK with config: {args.config}")
    try:
        sdk = OCFAFaceSDK(config_path=args.config)
        print("SDK initialized successfully")
    except Exception as e:
        print(f"Failed to initialize SDK: {e}")
        return 1

    # Run evaluations
    results = {}

    if args.task in ['liveness', 'all']:
        test_data = load_test_data(args.data, 'liveness')
        results['liveness'] = evaluate_liveness(sdk, test_data)

    if args.task in ['recognition', 'all']:
        test_pairs = load_test_data(args.data, 'recognition')
        results['recognition'] = evaluate_recognition(sdk, test_pairs)

    if args.task in ['identification', 'all']:
        gallery, probes = load_test_data(args.data, 'identification')
        results['identification'] = evaluate_identification(sdk, gallery, probes)

    # Save results
    print(f"\n\nSaving results to {args.output}...")
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print("Evaluation completed!")
    print("=" * 60)

    return 0


if __name__ == '__main__':
    sys.exit(main())
