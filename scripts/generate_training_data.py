"""
Generate Training Dataset from ADE20K → ROAD Mapping

This script:
1. Reads ade_to_road_mapping.json from annotation tool
2. Generates final masks using ADE → ROAD mapping
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
from typing import Dict, List, Tuple

import cv2
import numpy as np
from sklearn.model_selection import train_test_split


def load_ade_mapping(labeled_dir: Path) -> Dict[int, bool]:
    """Load ADE20K → ROAD mapping table."""
    mapping_path = labeled_dir / 'ade_to_road_mapping.json'
    
    if not mapping_path.exists():
        print(f"Warning: No mapping file found at {mapping_path}")
        return {}
    
    with open(mapping_path) as f:
        data = json.load(f)
        return {int(k): v for k, v in data.items()}


def generate_final_mask(
    session_dir: Path,
    image_stem: str,
    ade_mapping: Dict[int, bool]
) -> np.ndarray:
    """Generate final training mask using ADE → ROAD mapping."""
    ade_mask_path = session_dir / 'ade20k_masks' / (image_stem + '_ade20k.png')
    ade_mask = cv2.imread(str(ade_mask_path), cv2.IMREAD_GRAYSCALE)
    
    final_mask = np.zeros_like(ade_mask, dtype=np.uint8)
    
    for ade_id in np.unique(ade_mask):
        ade_id = int(ade_id)
        mask = (ade_mask == ade_id)
        
        if ade_mapping.get(ade_id, False):
            final_mask[mask] = 1  # ROAD
        else:
            final_mask[mask] = 2  # Obstacle
    
    return final_mask


def process_session(session_dir: Path, output_dir: Path) -> List[Tuple[Path, Path]]:
    """Process single session."""
    labeled_dir = session_dir / 'labeled'
    
    if not labeled_dir.exists():
        return []
    
    ade_mapping = load_ade_mapping(labeled_dir)
    
    if not ade_mapping:
        return []
    
    images_dir = output_dir / 'images'
    masks_dir = output_dir / 'masks'
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    
    raw_images_dir = session_dir.parent.parent.parent / 'raw_images' / session_dir.name
    image_files = sorted(list(raw_images_dir.glob('*.jpg')))
    
    pairs = []
    
    for img_file in image_files:
        final_mask = generate_final_mask(session_dir, img_file.stem, ade_mapping)
        
        output_img = images_dir / (session_dir.name + '_' + img_file.name)
        shutil.copy(img_file, output_img)
        
        output_mask = masks_dir / (session_dir.name + '_' + img_file.stem + '_mask.png')
        cv2.imwrite(str(output_mask), final_mask)
        
        pairs.append((output_img, output_mask))
    
    return pairs


def split_dataset(pairs: List[Tuple[Path, Path]], output_dir: Path, split_ratio: float = 0.8):
    """Split dataset into train/val."""
    train_pairs, val_pairs = train_test_split(pairs, train_size=split_ratio, random_state=42)
    
    train_img_dir = output_dir / 'images' / 'train'
    train_mask_dir = output_dir / 'masks' / 'train'
    val_img_dir = output_dir / 'images' / 'val'
    val_mask_dir = output_dir / 'masks' / 'val'
    
    for d in [train_img_dir, train_mask_dir, val_img_dir, val_mask_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    for img_path, mask_path in train_pairs:
        shutil.move(str(img_path), str(train_img_dir / img_path.name))
        shutil.move(str(mask_path), str(train_mask_dir / mask_path.name))
    
    for img_path, mask_path in val_pairs:
        shutil.move(str(img_path), str(val_img_dir / img_path.name))
        shutil.move(str(mask_path), str(val_mask_dir / mask_path.name))


def generate_statistics(output_dir: Path):
    """Generate dataset statistics."""
    stats = {
        'train': {'total': 0, 'background': 0, 'road': 0, 'obstacle': 0},
        'val': {'total': 0, 'background': 0, 'road': 0, 'obstacle': 0}
    }
    
    for split in ['train', 'val']:
        mask_dir = output_dir / 'masks' / split
        
        if not mask_dir.exists():
            continue
        
        for mask_file in mask_dir.glob('*.png'):
            mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
            
            stats[split]['total'] += 1
            stats[split]['background'] += int(np.sum(mask == 0))
            stats[split]['road'] += int(np.sum(mask == 1))
            stats[split]['obstacle'] += int(np.sum(mask == 2))
    
    for split in ['train', 'val']:
        total_pixels = stats[split]['background'] + stats[split]['road'] + stats[split]['obstacle']
        if total_pixels > 0:
            stats[split]['background_pct'] = stats[split]['background'] / total_pixels * 100
            stats[split]['road_pct'] = stats[split]['road'] / total_pixels * 100
            stats[split]['obstacle_pct'] = stats[split]['obstacle'] / total_pixels * 100
    
    stats_path = output_dir / 'statistics.json'
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\nDataset Statistics:")
    print(f"  Train: {stats['train']['total']} images")
    print(f"    Road:     {stats['train'].get('road_pct', 0):.1f}%")
    print(f"    Obstacle: {stats['train'].get('obstacle_pct', 0):.1f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sessions', type=str, nargs='+', required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--split', type=float, default=0.8)
    
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_pairs = []
    
    for session_pattern in args.sessions:
        session_dirs = list(Path('.').glob(session_pattern))
        
        for session_dir in session_dirs:
            if session_dir.name == 'labeled':
                session_dir = session_dir.parent
            
            pairs = process_session(session_dir, output_dir)
            all_pairs.extend(pairs)
    
    if not all_pairs:
        print("Error: No data found!")
        return 1
    
    split_dataset(all_pairs, output_dir, args.split)
    generate_statistics(output_dir)
    
    print("\nDataset Generation Complete!")
    
    return 0


if __name__ == '__main__':
    exit(main())
