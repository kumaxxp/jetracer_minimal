"""
Prepare dataset for training by splitting into train/val sets.

This script:
1. Matches images with their masks
2. Splits into training and validation sets
3. Creates dataset metadata
4. Generates data augmentation preview (optional)

Usage:
    python scripts/prepare_dataset.py \
        --images data/raw_images \
        --masks data/annotations/masks/masks \
        --output data/datasets/road_segmentation \
        --split 0.8 \
        --visualize
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import List, Tuple, Dict, Any

import cv2
import numpy as np


def find_image_mask_pairs(
    images_dir: Path,
    masks_dir: Path
) -> List[Tuple[Path, Path]]:
    """
    Find matching image-mask pairs.
    
    Args:
        images_dir: Directory containing images
        masks_dir: Directory containing masks
        
    Returns:
        List of (image_path, mask_path) tuples
    """
    pairs = []
    
    # Find all mask files
    mask_files = list(masks_dir.glob('*_mask.png'))
    print(f"Found {len(mask_files)} mask files")
    
    for mask_path in mask_files:
        # Derive image filename from mask filename
        # img_0001_mask.png -> img_0001.jpg
        image_name = mask_path.stem.replace('_mask', '') + '.jpg'
        
        # Search for image in all session directories
        image_path = None
        
        # First try direct path
        candidate = images_dir / image_name
        if candidate.exists():
            image_path = candidate
        else:
            # Search in session directories
            for session_dir in images_dir.glob('session_*'):
                candidate = session_dir / image_name
                if candidate.exists():
                    image_path = candidate
                    break
        
        if image_path is not None:
            pairs.append((image_path, mask_path))
        else:
            print(f"  Warning: Image not found for mask: {mask_path.name}")
    
    print(f"Found {len(pairs)} valid image-mask pairs")
    return pairs


def split_dataset(
    pairs: List[Tuple[Path, Path]],
    train_ratio: float = 0.8,
    seed: int = 42
) -> Tuple[List[Tuple[Path, Path]], List[Tuple[Path, Path]]]:
    """
    Split dataset into training and validation sets.
    
    Args:
        pairs: List of (image, mask) path tuples
        train_ratio: Ratio of training samples (0.0-1.0)
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_pairs, val_pairs)
    """
    # Set random seed
    random.seed(seed)
    
    # Shuffle pairs
    pairs_shuffled = pairs.copy()
    random.shuffle(pairs_shuffled)
    
    # Split
    split_idx = int(len(pairs_shuffled) * train_ratio)
    train_pairs = pairs_shuffled[:split_idx]
    val_pairs = pairs_shuffled[split_idx:]
    
    print(f"Split: {len(train_pairs)} train, {len(val_pairs)} val")
    
    return train_pairs, val_pairs


def copy_dataset(
    pairs: List[Tuple[Path, Path]],
    output_dir: Path,
    split_name: str
) -> None:
    """
    Copy image-mask pairs to output directory.
    
    Args:
        pairs: List of (image, mask) path tuples
        output_dir: Output directory
        split_name: 'train' or 'val'
    """
    # Create directories
    images_dir = output_dir / split_name / 'images'
    masks_dir = output_dir / split_name / 'masks'
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Copying {split_name} set...")
    
    for idx, (image_path, mask_path) in enumerate(pairs):
        # Use simple sequential naming for dataset
        new_name = f"{split_name}_{idx:04d}"
        
        # Copy image
        shutil.copy2(image_path, images_dir / f"{new_name}.jpg")
        
        # Copy mask
        shutil.copy2(mask_path, masks_dir / f"{new_name}.png")
    
    print(f"  Copied {len(pairs)} pairs to {split_name}/")


def analyze_dataset(
    pairs: List[Tuple[Path, Path]]
) -> Dict[str, Any]:
    """
    Analyze dataset statistics.
    
    Args:
        pairs: List of (image, mask) path tuples
        
    Returns:
        Dictionary of statistics
    """
    stats = {
        'num_samples': len(pairs),
        'class_distribution': {0: 0, 1: 0, 2: 0},
        'image_sizes': [],
        'mask_pixel_counts': {0: 0, 1: 0, 2: 0}
    }
    
    print("Analyzing dataset...")
    
    for idx, (image_path, mask_path) in enumerate(pairs):
        # Load mask
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        
        # Image size
        stats['image_sizes'].append(mask.shape)
        
        # Class distribution (images containing each class)
        unique_classes = np.unique(mask)
        for class_id in unique_classes:
            if class_id in stats['class_distribution']:
                stats['class_distribution'][class_id] += 1
        
        # Pixel counts
        for class_id in [0, 1, 2]:
            pixel_count = np.sum(mask == class_id)
            stats['mask_pixel_counts'][class_id] += pixel_count
        
        if (idx + 1) % 10 == 0:
            print(f"  Analyzed {idx + 1}/{len(pairs)} samples")
    
    # Calculate percentages
    total_pixels = sum(stats['mask_pixel_counts'].values())
    stats['class_percentages'] = {
        class_id: (count / total_pixels * 100) if total_pixels > 0 else 0
        for class_id, count in stats['mask_pixel_counts'].items()
    }
    
    return stats


def create_visualization_grid(
    pairs: List[Tuple[Path, Path]],
    output_path: Path,
    num_samples: int = 6
) -> None:
    """
    Create visualization grid of sample images and masks.
    
    Args:
        pairs: List of (image, mask) path tuples
        output_path: Output path for visualization
        num_samples: Number of samples to show
    """
    # Select random samples
    sample_pairs = random.sample(pairs, min(num_samples, len(pairs)))
    
    # Define colors for each class
    colors = {
        0: (128, 128, 128),  # Background - Gray
        1: (0, 255, 0),      # Road - Green
        2: (0, 0, 255),      # Obstacle - Red
    }
    
    # Create grid (2 rows per sample: image + mask overlay)
    rows = []
    for image_path, mask_path in sample_pairs:
        # Load image and mask
        image = cv2.imread(str(image_path))
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        
        if image is None or mask is None:
            continue
        
        # Resize for visualization (smaller)
        scale = 0.5
        h, w = image.shape[:2]
        new_h, new_w = int(h * scale), int(w * scale)
        image_small = cv2.resize(image, (new_w, new_h))
        mask_small = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        
        # Create colored mask overlay
        colored_mask = np.zeros_like(image_small)
        for class_id, color in colors.items():
            colored_mask[mask_small == class_id] = color
        
        overlay = cv2.addWeighted(image_small, 0.6, colored_mask, 0.4, 0)
        
        # Stack image and overlay vertically
        pair_vis = np.vstack([image_small, overlay])
        rows.append(pair_vis)
    
    # Create grid (2 columns)
    col1 = np.vstack(rows[::2]) if len(rows) > 1 else rows[0]
    col2 = np.vstack(rows[1::2]) if len(rows) > 2 else rows[1] if len(rows) > 1 else col1
    
    # Ensure same height
    if col1.shape[0] != col2.shape[0]:
        min_height = min(col1.shape[0], col2.shape[0])
        col1 = col1[:min_height]
        col2 = col2[:min_height]
    
    grid = np.hstack([col1, col2])
    
    # Save
    cv2.imwrite(str(output_path), grid)
    print(f"Visualization saved to: {output_path}")


def save_metadata(
    output_dir: Path,
    train_stats: Dict[str, Any],
    val_stats: Dict[str, Any],
    config: Dict[str, Any]
) -> None:
    """
    Save dataset metadata.
    
    Args:
        output_dir: Output directory
        train_stats: Training set statistics
        val_stats: Validation set statistics
        config: Configuration dictionary
    """
    metadata = {
        'dataset_name': output_dir.name,
        'created_at': str(Path.cwd()),
        'config': config,
        'train': train_stats,
        'val': val_stats,
        'class_names': {
            0: 'Background',
            1: 'Road',
            2: 'Obstacle'
        }
    }
    
    metadata_path = output_dir / 'dataset_metadata.json'
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, default=str)
    
    print(f"Metadata saved to: {metadata_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Prepare segmentation dataset for training'
    )
    parser.add_argument(
        '--images',
        type=str,
        required=True,
        help='Directory containing images'
    )
    parser.add_argument(
        '--masks',
        type=str,
        required=True,
        help='Directory containing masks'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output directory for prepared dataset'
    )
    parser.add_argument(
        '--split',
        type=float,
        default=0.8,
        help='Train/val split ratio (default: 0.8)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Create visualization grid'
    )
    
    args = parser.parse_args()
    
    # Convert paths
    images_dir = Path(args.images)
    masks_dir = Path(args.masks)
    output_dir = Path(args.output)
    
    # Validate inputs
    if not images_dir.exists():
        print(f"Error: Images directory not found: {images_dir}")
        return 1
    
    if not masks_dir.exists():
        print(f"Error: Masks directory not found: {masks_dir}")
        return 1
    
    print("=" * 60)
    print("Dataset Preparation")
    print("=" * 60)
    print(f"Images: {images_dir}")
    print(f"Masks: {masks_dir}")
    print(f"Output: {output_dir}")
    print(f"Train/Val split: {args.split:.2f}/{1-args.split:.2f}")
    print(f"Random seed: {args.seed}")
    print("=" * 60)
    print()
    
    # Find image-mask pairs
    pairs = find_image_mask_pairs(images_dir, masks_dir)
    
    if len(pairs) == 0:
        print("Error: No valid image-mask pairs found!")
        return 1
    
    # Split dataset
    train_pairs, val_pairs = split_dataset(pairs, args.split, args.seed)
    
    # Copy to output directory
    copy_dataset(train_pairs, output_dir, 'train')
    copy_dataset(val_pairs, output_dir, 'val')
    
    # Analyze datasets
    print()
    train_stats = analyze_dataset(train_pairs)
    val_stats = analyze_dataset(val_pairs)
    
    # Save metadata
    config = {
        'images_dir': str(images_dir),
        'masks_dir': str(masks_dir),
        'split_ratio': args.split,
        'random_seed': args.seed
    }
    save_metadata(output_dir, train_stats, val_stats, config)
    
    # Create visualization
    if args.visualize and len(train_pairs) > 0:
        vis_path = output_dir / 'dataset_preview.jpg'
        create_visualization_grid(train_pairs, vis_path)
    
    # Print summary
    print()
    print("=" * 60)
    print("Dataset Preparation Complete!")
    print("=" * 60)
    print(f"Total samples: {len(pairs)}")
    print(f"  Training: {len(train_pairs)}")
    print(f"  Validation: {len(val_pairs)}")
    print()
    print("Class distribution (Training):")
    for class_id, count in train_stats['class_distribution'].items():
        class_name = ['Background', 'Road', 'Obstacle'][class_id]
        percentage = train_stats['class_percentages'][class_id]
        print(f"  {class_name}: {count} images ({percentage:.1f}% pixels)")
    print()
    print(f"Dataset saved to: {output_dir}")
    print("=" * 60)
    
    return 0


if __name__ == '__main__':
    exit(main())
