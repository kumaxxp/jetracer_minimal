"""
Depth-based automatic annotation for JetRacer.

Uses depth estimation to identify floor (drivable) vs obstacles.
This approach is specifically designed for low-viewpoint robot cameras.

Usage:
    python scripts/depth_annotate.py \
        --input data/raw_images \
        --output data/annotations/depth_masks \
        --visualize
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple, Dict

import cv2
import numpy as np
import torch
from PIL import Image


try:
    from transformers import DPTImageProcessor, DPTForDepthEstimation
    HAS_DPT = True
except ImportError:
    HAS_DPT = False


CLASS_COLORS = {
    0: (128, 128, 128),  # Background - Gray
    1: (0, 255, 0),      # Road - Green
    2: (0, 0, 255),      # Obstacle - Red
}


class DepthAnnotator:
    """Depth-based segmentation for robot navigation."""
    
    def __init__(self, model_name: str = "Intel/dpt-large"):
        """
        Initialize depth estimation model.
        
        Args:
            model_name: HuggingFace model name for depth estimation
        """
        if not HAS_DPT:
            raise ImportError("transformers not installed. Run: pip install transformers")
        
        print(f"Loading depth model: {model_name}")
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load model
        self.processor = DPTImageProcessor.from_pretrained(model_name)
        self.model = DPTForDepthEstimation.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        print("✓ Model loaded successfully")
    
    def estimate_depth(self, image: np.ndarray) -> np.ndarray:
        """
        Estimate depth map from RGB image.
        
        Args:
            image: RGB image (H, W, 3)
            
        Returns:
            Depth map (H, W) - normalized to [0, 1]
        """
        # Convert to PIL
        pil_image = Image.fromarray(image)
        
        # Process
        inputs = self.processor(images=pil_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            predicted_depth = outputs.predicted_depth
        
        # Interpolate to original size
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=image.shape[:2],
            mode="bicubic",
            align_corners=False,
        )
        
        # Convert to numpy and normalize
        depth = prediction.squeeze().cpu().numpy()
        depth = (depth - depth.min()) / (depth.max() - depth.min())
        
        return depth
    
    def depth_to_segmentation(
        self,
        depth: np.ndarray,
        image: np.ndarray,
        floor_threshold: float = 0.6,
        gradient_threshold: float = 0.15
    ) -> Tuple[np.ndarray, Dict]:
        """
        Convert depth map to segmentation mask.
        
        Strategy for low-viewpoint robot:
        - Floor (Road): Near regions with low gradient (horizontal surfaces)
        - Obstacle: Far regions OR high gradient (vertical surfaces, objects)
        - Background: Very far regions (ceiling, distant walls)
        
        Args:
            depth: Depth map (H, W), normalized [0, 1]
            image: Original image (for texture analysis)
            floor_threshold: Depth threshold for floor detection
            gradient_threshold: Gradient threshold for vertical surfaces
            
        Returns:
            Tuple of (mask, statistics)
        """
        h, w = depth.shape
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # Compute depth gradient (vertical surfaces have high gradient)
        grad_y = np.abs(cv2.Sobel(depth, cv2.CV_64F, 0, 1, ksize=3))
        grad_x = np.abs(cv2.Sobel(depth, cv2.CV_64F, 1, 0, ksize=3))
        gradient = np.sqrt(grad_x**2 + grad_y**2)
        gradient = (gradient - gradient.min()) / (gradient.max() - gradient.min() + 1e-8)
        
        # Strategy: Bottom region is typically floor for robot cameras
        height_weight = np.linspace(0, 1, h).reshape(-1, 1)  # 0 at top, 1 at bottom
        height_weight = np.tile(height_weight, (1, w))
        
        # Floor detection: Near depth + Low gradient + Bottom region
        floor_score = (
            (depth > floor_threshold) * 0.4 +          # Near regions
            (gradient < gradient_threshold) * 0.3 +     # Low gradient (horizontal)
            (height_weight > 0.4) * 0.3                 # Bottom half of image
        )
        
        # Obstacle detection: High gradient OR (medium depth + top region)
        obstacle_score = (
            (gradient > gradient_threshold) * 0.6 +     # High gradient (vertical)
            ((depth < floor_threshold) * (height_weight < 0.6)) * 0.4  # Far + upper region
        )
        
        # Apply thresholds
        mask[floor_score > 0.6] = 1      # Road
        mask[obstacle_score > 0.5] = 2   # Obstacle
        # Rest remains 0 (Background)
        
        # Post-processing: Clean up small regions
        mask = self._morphological_cleanup(mask)
        
        # Statistics
        stats = {
            'total_pixels': mask.size,
            'background': int(np.sum(mask == 0)),
            'road': int(np.sum(mask == 1)),
            'obstacle': int(np.sum(mask == 2)),
        }
        
        stats['background_pct'] = stats['background'] / stats['total_pixels'] * 100
        stats['road_pct'] = stats['road'] / stats['total_pixels'] * 100
        stats['obstacle_pct'] = stats['obstacle'] / stats['total_pixels'] * 100
        
        return mask, stats
    
    def _morphological_cleanup(self, mask: np.ndarray) -> np.ndarray:
        """Clean up mask using morphological operations."""
        # Remove small noise
        kernel = np.ones((5, 5), np.uint8)
        
        # For each class
        for class_id in [1, 2]:
            class_mask = (mask == class_id).astype(np.uint8)
            
            # Morphological opening (remove small noise)
            class_mask = cv2.morphologyEx(class_mask, cv2.MORPH_OPEN, kernel, iterations=1)
            
            # Morphological closing (fill small holes)
            class_mask = cv2.morphologyEx(class_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
            
            # Apply back
            mask[class_mask > 0] = class_id
        
        return mask
    
    def annotate_image(self, image_path: Path) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Generate annotation for a single image.
        
        Args:
            image_path: Path to input image
            
        Returns:
            Tuple of (image, mask, statistics)
        """
        print(f"Processing: {image_path.name}")
        
        # Load image
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Estimate depth
        print("  Estimating depth...")
        depth = self.estimate_depth(image)
        
        # Convert to segmentation
        print("  Converting to segmentation...")
        mask, stats = self.depth_to_segmentation(depth, image)
        
        print(f"  Distribution: Background={stats['background_pct']:.1f}%, "
              f"Road={stats['road_pct']:.1f}%, Obstacle={stats['obstacle_pct']:.1f}%")
        
        return image, mask, stats
    
    def create_visualization(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        alpha: float = 0.5
    ) -> np.ndarray:
        """Create visualization."""
        colored_mask = np.zeros_like(image)
        for class_id, color in CLASS_COLORS.items():
            colored_mask[mask == class_id] = color
        
        vis = cv2.addWeighted(image, 1 - alpha, colored_mask, alpha, 0)
        return vis


def process_images(
    annotator: DepthAnnotator,
    input_dir: Path,
    output_dir: Path,
    visualize: bool = False
) -> Dict:
    """Process all images."""
    # Create output directories
    masks_dir = output_dir / 'masks'
    masks_dir.mkdir(parents=True, exist_ok=True)
    
    if visualize:
        vis_dir = output_dir / 'visualizations'
        vis_dir.mkdir(parents=True, exist_ok=True)
    
    # Find images
    image_files = []
    for session_dir in input_dir.glob('session_*'):
        image_files.extend(list(session_dir.glob('*.jpg')))
    
    if not image_files:
        image_files = list(input_dir.glob('*.jpg'))
    
    print(f"Found {len(image_files)} images\n")
    
    # Statistics
    stats = {
        'total': len(image_files),
        'processed': 0,
        'failed': 0
    }
    
    # Process each image
    for idx, image_path in enumerate(image_files):
        try:
            # Annotate
            image, mask, img_stats = annotator.annotate_image(image_path)
            
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
            
            if (idx + 1) % 5 == 0:
                print(f"Processed {idx + 1}/{len(image_files)} images")
            
        except Exception as e:
            print(f"Error processing {image_path.name}: {e}")
            stats['failed'] += 1
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description='Depth-based automatic annotation'
    )
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--model', type=str, default='Intel/dpt-large')
    parser.add_argument('--visualize', action='store_true')
    
    args = parser.parse_args()
    
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    
    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        return 1
    
    print("=" * 60)
    print("Depth-Based Auto-Annotation")
    print("=" * 60)
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Model: {args.model}")
    print("=" * 60)
    print()
    
    # Initialize annotator
    annotator = DepthAnnotator(args.model)
    
    # Process images
    print()
    stats = process_images(annotator, input_dir, output_dir, args.visualize)
    
    # Summary
    print()
    print("=" * 60)
    print("Depth-Based Annotation Complete!")
    print("=" * 60)
    print(f"Total: {stats['total']}")
    print(f"Processed: {stats['processed']}")
    print(f"Failed: {stats['failed']}")
    print()
    print(f"Masks: {output_dir / 'masks'}")
    if args.visualize:
        print(f"Visualizations: {output_dir / 'visualizations'}")
    print("=" * 60)
    
    return 0


if __name__ == '__main__':
    exit(main())
