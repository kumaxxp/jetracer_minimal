"""
Automatic annotation using OneFormer (ADE20K).

This script:
1. Loads pre-trained OneFormer model
2. Processes collected images
3. Maps ADE20K classes to JetRacer classes
4. Saves segmentation masks

Usage:
    python scripts/auto_annotate.py \
        --input data/raw_images \
        --output data/annotations/auto_masks \
        --visualize
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation


# ADE20K class mapping to JetRacer classes
# ADE20K: 150 classes (0-149)
# JetRacer: 3 classes (0: Background, 1: Road, 2: Obstacle)
ADE20K_TO_JETRACER = {
    # Road-like surfaces (→ Road: 1)
    'floor': 1,
    'road': 1,
    'sidewalk, pavement': 1,
    'path': 1,
    'runway': 1,
    'carpet, carpeting, rug': 1,
    'floor, flooring': 1,
    
    # Obstacles (→ Obstacle: 2)
    'wall': 2,
    'building, edifice': 2,
    'house': 2,
    'fence, fencing': 2,
    'door, double door': 2,
    'box': 2,
    'chair': 2,
    'table': 2,
    'cabinet': 2,
    'person, individual, someone': 2,
    'car, auto, automobile': 2,
    'stairs, steps': 2,
    'stairway, staircase': 2,
    'step, stair': 2,
    'column, pillar': 2,
    'signboard, sign': 2,
    'pole': 2,
    'trade name, brand name': 2,
    
    # Background (→ Background: 0)
    'ceiling': 0,
    'sky': 0,
    'wall, brick wall': 0,
    'light, light source': 0,
    'window': 0,
    'windowpane, window': 0,
}


# Colors for visualization
CLASS_COLORS = {
    0: (128, 128, 128),  # Background - Gray
    1: (0, 255, 0),      # Road - Green
    2: (0, 0, 255),      # Obstacle - Red
}


class AutoAnnotator:
    """Automatic annotation using OneFormer."""
    
    def __init__(self, model_name: str = "shi-labs/oneformer_ade20k_swin_tiny"):
        """
        Initialize OneFormer model.
        
        Args:
            model_name: HuggingFace model name
        """
        print(f"Loading model: {model_name}")
        print("This may take a few minutes on first run...")
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load processor and model
        self.processor = OneFormerProcessor.from_pretrained(model_name)
        self.model = OneFormerForUniversalSegmentation.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        # Get ADE20K class names
        self.ade20k_classes = self.processor.tokenizer.get_vocab()
        self.id2label = {v: k for k, v in self.ade20k_classes.items()}
        
        print("✓ Model loaded successfully")
    
    def map_ade20k_to_jetracer(self, ade20k_mask: np.ndarray) -> np.ndarray:
        """
        Map ADE20K classes to JetRacer classes.
        
        Args:
            ade20k_mask: Segmentation mask with ADE20K class IDs
            
        Returns:
            Mask with JetRacer class IDs (0: Background, 1: Road, 2: Obstacle)
        """
        jetracer_mask = np.zeros_like(ade20k_mask, dtype=np.uint8)
        
        for ade20k_id in np.unique(ade20k_mask):
            if ade20k_id >= len(self.id2label):
                continue
            
            # Get class name
            class_name = self.id2label.get(ade20k_id, '')
            
            # Map to JetRacer class
            jetracer_class = ADE20K_TO_JETRACER.get(class_name, 0)  # Default: Background
            
            # Apply mapping
            jetracer_mask[ade20k_mask == ade20k_id] = jetracer_class
        
        return jetracer_mask
    
    def annotate_image(self, image_path: Path) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate annotation for a single image.
        
        Args:
            image_path: Path to input image
            
        Returns:
            Tuple of (original_image, jetracer_mask)
        """
        # Load image
        image = Image.open(image_path).convert('RGB')
        image_np = np.array(image)
        
        # Prepare inputs
        inputs = self.processor(
            images=image,
            task_inputs=["semantic"],
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Inference
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Post-process
        predicted_semantic_map = self.processor.post_process_semantic_segmentation(
            outputs, target_sizes=[image.size[::-1]]
        )[0]
        
        # Convert to numpy
        ade20k_mask = predicted_semantic_map.cpu().numpy().astype(np.uint8)
        
        # Map to JetRacer classes
        jetracer_mask = self.map_ade20k_to_jetracer(ade20k_mask)
        
        return image_np, jetracer_mask
    
    def create_visualization(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        alpha: float = 0.5
    ) -> np.ndarray:
        """
        Create visualization by overlaying mask on image.
        
        Args:
            image: Original image (H, W, 3)
            mask: Segmentation mask (H, W)
            alpha: Overlay transparency
            
        Returns:
            Visualization image (H, W, 3)
        """
        # Create colored mask
        colored_mask = np.zeros_like(image)
        for class_id, color in CLASS_COLORS.items():
            colored_mask[mask == class_id] = color
        
        # Blend with original image
        vis = cv2.addWeighted(image, 1 - alpha, colored_mask, alpha, 0)
        
        return vis


def process_images(
    annotator: AutoAnnotator,
    input_dir: Path,
    output_dir: Path,
    visualize: bool = False
) -> Dict[str, int]:
    """
    Process all images in input directory.
    
    Args:
        annotator: AutoAnnotator instance
        input_dir: Input directory
        output_dir: Output directory
        visualize: Whether to create visualizations
        
    Returns:
        Processing statistics
    """
    # Create output directories
    masks_dir = output_dir / 'masks'
    masks_dir.mkdir(parents=True, exist_ok=True)
    
    if visualize:
        vis_dir = output_dir / 'visualizations'
        vis_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all images
    image_files = []
    for session_dir in input_dir.glob('session_*'):
        image_files.extend(list(session_dir.glob('*.jpg')))
    
    if not image_files:
        # Try direct path
        image_files = list(input_dir.glob('*.jpg'))
    
    print(f"Found {len(image_files)} images")
    
    # Statistics
    stats = {
        'total': len(image_files),
        'processed': 0,
        'failed': 0
    }
    
    # Process each image
    for idx, image_path in enumerate(image_files):
        try:
            print(f"Processing [{idx+1}/{len(image_files)}]: {image_path.name}")
            
            # Annotate
            image, mask = annotator.annotate_image(image_path)
            
            # Save mask
            mask_filename = image_path.stem + '_mask.png'
            mask_path = masks_dir / mask_filename
            cv2.imwrite(str(mask_path), mask)
            
            # Create visualization
            if visualize:
                vis = annotator.create_visualization(image, mask)
                vis_filename = image_path.stem + '_vis.jpg'
                vis_path = vis_dir / vis_filename
                cv2.imwrite(str(vis_path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
            
            stats['processed'] += 1
            
        except Exception as e:
            print(f"  Error: {e}")
            stats['failed'] += 1
            import traceback
            traceback.print_exc()
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description='Automatic annotation using OneFormer'
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input directory containing images'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/annotations/auto_masks',
        help='Output directory for masks'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='shi-labs/oneformer_ade20k_swin_tiny',
        help='OneFormer model name'
    )
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Create visualization images'
    )
    
    args = parser.parse_args()
    
    # Convert paths
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    
    # Validate input
    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        return 1
    
    print("=" * 60)
    print("Automatic Annotation with OneFormer")
    print("=" * 60)
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Model: {args.model}")
    print(f"Visualize: {args.visualize}")
    print("=" * 60)
    print()
    
    # Initialize annotator
    annotator = AutoAnnotator(args.model)
    
    # Process images
    print()
    stats = process_images(annotator, input_dir, output_dir, args.visualize)
    
    # Print statistics
    print()
    print("=" * 60)
    print("Auto-Annotation Complete!")
    print("=" * 60)
    print(f"Total images: {stats['total']}")
    print(f"Successfully processed: {stats['processed']}")
    print(f"Failed: {stats['failed']}")
    print()
    print(f"Masks saved to: {output_dir / 'masks'}")
    if args.visualize:
        print(f"Visualizations saved to: {output_dir / 'visualizations'}")
    print("=" * 60)
    
    return 0


if __name__ == '__main__':
    exit(main())
