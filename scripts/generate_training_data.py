"""
Generate Training Dataset from Annotated Metadata

This script:
1. Reads metadata.json from annotation tool
2. Generates final masks where ignorable objects are treated as Road
3. Creates train/val split
4. Prepares dataset for training

Usage:
    python scripts/generate_training_data.py \
        --sessions data/annotations/oneformer_decisive/*/labeled \
        --output data/datasets/jetracer_final \
        --split 0.8
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from sklearn.model_selection import train_test_split


def load_metadata(labeled_dir: Path) -> dict:
    """Load metadata.json from labeled directory."""
    metadata_path = labeled_dir / 'metadata.json'
    with open(metadata_path) as f:
        return json.load(f)


def generate_final_mask(
    session_dir: Path,
    image_stem: str,
    metadata: dict
) -> np.ndarray:
    """
    Generate final training mask.
    
    Strategy:
    - Road (class 1): Keep as is + add ignorable objects
    - Obstacle (class 2): Only enabled objects
    - Background (class 0): Everything else
    """
    # Load original JetRacer mask
    jr_mask_path = session_dir / 'masks' / (image_stem + '_mask.png')
    jr_mask = cv2.imread(str(jr_mask_path), cv2.IMREAD_GRAYSCALE)
    
    # Load ADE mask
    ade_mask_path = session_dir / 'ade20k_masks' / (image_stem + '_ade20k.png')
    ade_mask = cv2.imread(str(ade_mask_path), cv2.IMREAD_GRAYSCALE)
    
    # Start with original mask
    final_mask = jr_mask.copy()
    
    # Get object metadata
    if image_stem not in metadata:
        return final_mask
    
    obj_metadata = metadata[image_stem]['objects']
    
    # Process each object
    for obj_id, enabled in obj_metadata.items():
        if obj_id.startswith('ade_'):
            ade_id = int(obj_id.split('_')[1])
            obj_pixels = (ade_mask == ade_id) & (jr_mask != 1)
            
            if not enabled:
                # Ignorable object → treat as Road
                final_mask[obj_pixels] = 1
    
    return final_mask


def process_session(
    session_dir: Path,
    output_dir: Path
) -> List[Tuple[Path, Path]]:
    """
    Process single session.
    
    Returns:
        List of (image_path, mask_path) tuples
    """
    labeled_dir = session_dir / 'labeled'
    
    if not labeled_dir.exists():
        print(f"Warning: No labeled directory in {session_dir.name}")
        return []
    
    # Load metadata
    metadata = load_metadata(labeled_dir)
    
    # Output directories
    images_dir = output_dir / 'images'
    masks_dir = output_dir / 'masks'
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    
    # Find images
    raw_images_dir = session_dir.parent.parent.parent / 'raw_images' / session_dir.name
    image_files = sorted(list(raw_images_dir.glob('*.jpg')))
    
    pairs = []
    
    for img_file in image_files:
        # Generate final mask
        final_mask = generate_final_mask(session_dir, img_file.stem, metadata)
        
        # Copy image
        output_img = images_dir / (session_dir.name + '_' + img_file.name)
        shutil.copy(img_file, output_img)
        
        # Save mask
        output_mask = masks_dir / (session_dir.name + '_' + img_file.stem + '_mask.png')
        cv2.imwrite(str(output_mask), final_mask)
        
        pairs.append((output_img, output_mask))
        
        print(f"  ✓ {img_file.name}")
    
    return pairs


def create_dataset_yaml(output_dir: Path, num_classes: int = 3):
    """Create dataset.yaml for training."""
    yaml_content = f"""# JetRacer Dataset
path: {output_dir.absolute()}
train: images/train
val: images/val

# Classes
nc: {num_classes}
names: ['Background', 'Road', 'Obstacle']

# Notes:
# - Road: Passable surfaces (including ignorable objects)
# - Obstacle: Objects to avoid
# - Background: Sky, ceiling, far objects
"""
    
    yaml_path = output_dir / 'dataset.yaml'
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"Created dataset.yaml: {yaml_path}")


def split_dataset(
    pairs: List[Tuple[Path, Path]],
    output_dir: Path,
    split_ratio: float = 0.8
):
    """Split dataset into train/val."""
    # Split
    train_pairs, val_pairs = train_test_split(
        pairs,
        train_size=split_ratio,
        random_state=42
    )
    
    print(f"\nSplit: {len(train_pairs)} train, {len(val_pairs)} val")
    
    # Create directories
    train_img_dir = output_dir / 'images' / 'train'
    train_mask_dir = output_dir / 'masks' / 'train'
    val_img_dir = output_dir / 'images' / 'val'
    val_mask_dir = output_dir / 'masks' / 'val'
    
    for d in [train_img_dir, train_mask_dir, val_img_dir, val_mask_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    # Move files
    for img_path, mask_path in train_pairs:
        shutil.move(img_path, train_img_dir / img_path.name)
        shutil.move(mask_path, train_mask_dir / mask_path.name)
    
    for img_path, mask_path in val_pairs:
        shutil.move(img_path, val_img_dir / img_path.name)
        shutil.move(mask_path, val_mask_dir / mask_path.name)
    
    # Remove temporary directories
    (output_dir / 'images' / 'temp').rmdir() if (output_dir / 'images' / 'temp').exists() else None
    (output_dir / 'masks' / 'temp').rmdir() if (output_dir / 'masks' / 'temp').exists() else None


def generate_statistics(output_dir: Path):
    """Generate dataset statistics."""
    stats = {
        'train': {'total': 0, 'background': 0, 'road': 0, 'obstacle': 0},
        'val': {'total': 0, 'background': 0, 'road': 0, 'obstacle': 0}
    }
    
    for split in ['train', 'val']:
        mask_dir = output_dir / 'masks' / split
        
        for mask_file in mask_dir.glob('*.png'):
            mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
            
            stats[split]['total'] += 1
            stats[split]['background'] += np.sum(mask == 0)
            stats[split]['road'] += np.sum(mask == 1)
            stats[split]['obstacle'] += np.sum(mask == 2)
    
    # Calculate percentages
    for split in ['train', 'val']:
        total_pixels = stats[split]['background'] + stats[split]['road'] + stats[split]['obstacle']
        if total_pixels > 0:
            stats[split]['background_pct'] = stats[split]['background'] / total_pixels * 100
            stats[split]['road_pct'] = stats[split]['road'] / total_pixels * 100
            stats[split]['obstacle_pct'] = stats[split]['obstacle'] / total_pixels * 100
    
    # Save statistics
    stats_path = output_dir / 'statistics.json'
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\nDataset Statistics:")
    print(f"  Train: {stats['train']['total']} images")
    print(f"    Background: {stats['train'].get('background_pct', 0):.1f}%")
    print(f"    Road:       {stats['train'].get('road_pct', 0):.1f}%")
    print(f"    Obstacle:   {stats['train'].get('obstacle_pct', 0):.1f}%")
    print(f"  Val: {stats['val']['total']} images")
    print(f"    Background: {stats['val'].get('background_pct', 0):.1f}%")
    print(f"    Road:       {stats['val'].get('road_pct', 0):.1f}%")
    print(f"    Obstacle:   {stats['val'].get('obstacle_pct', 0):.1f}%")


def main():
    parser = argparse.ArgumentParser(
        description='Generate training dataset from annotated metadata'
    )
    parser.add_argument('--sessions', type=str, nargs='+', required=True,
                       help='Paths to OneFormer session directories with labeled subdirectory')
    parser.add_argument('--output', type=str, required=True,
                       help='Output directory for dataset')
    parser.add_argument('--split', type=float, default=0.8,
                       help='Train/val split ratio (default: 0.8)')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Generate Training Dataset from Annotated Metadata")
    print("=" * 60)
    print(f"Output: {output_dir}")
    print("=" * 60)
    print()
    
    # Process all sessions
    all_pairs = []
    
    for session_pattern in args.sessions:
        session_dirs = list(Path('.').glob(session_pattern))
        
        for session_dir in session_dirs:
            # Get parent (oneformer_decisive/session_XXX)
            if session_dir.name == 'labeled':
                session_dir = session_dir.parent
            
            print(f"Processing: {session_dir.name}")
            pairs = process_session(session_dir, output_dir)
            all_pairs.extend(pairs)
    
    if not all_pairs:
        print("Error: No data found!")
        return 1
    
    print(f"\nTotal: {len(all_pairs)} image-mask pairs")
    
    # Split dataset
    split_dataset(all_pairs, output_dir, args.split)
    
    # Create dataset.yaml
    create_dataset_yaml(output_dir)
    
    # Generate statistics
    generate_statistics(output_dir)
    
    print("\n" + "=" * 60)
    print("Dataset Generation Complete!")
    print("=" * 60)
    print(f"Location: {output_dir}")
    print(f"Train images: {output_dir}/images/train/")
    print(f"Val images: {output_dir}/images/val/")
    print(f"Config: {output_dir}/dataset.yaml")
    print("=" * 60)
    
    return 0


if __name__ == '__main__':
    exit(main())
