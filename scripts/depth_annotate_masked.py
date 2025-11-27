"""
Depth-based annotation with vehicle body masking for JetRacer.

Key improvement: Masks out the vehicle body that appears in the bottom portion
of every image, which was being incorrectly classified as "Road".

Usage:
    python scripts/depth_annotate_masked.py \
        --input data/raw_images \
        --output data/annotations/depth_masks_masked \
        --vehicle-mask-bottom 0.25 \
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
    3: (64, 64, 64),     # Vehicle body - Dark Gray (optional visualization)
}


class MaskedDepthAnnotator:
    """Depth-based segmentation with vehicle body masking."""
    
    def __init__(
        self,
        model_name: str = "Intel/dpt-large",
        vehicle_mask_bottom: float = 0.25,  # Mask bottom 25% by default
        vehicle_mask_sides: float = 0.0     # Optional: mask left/right edges
    ):
        """
        Initialize with vehicle masking parameters.
        
        Args:
            model_name: Depth estimation model
            vehicle_mask_bottom: Fraction of image height to mask from bottom (0.0-1.0)
            vehicle_mask_sides: Fraction of image width to mask from each side (0.0-0.5)
        """
        if not HAS_DPT:
            raise ImportError("transformers not installed")
        
        print(f"Loading depth model: {model_name}")
        print(f"Vehicle mask: bottom {vehicle_mask_bottom*100:.0f}%, sides {vehicle_mask_sides*100:.0f}%")
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self.processor = DPTImageProcessor.from_pretrained(model_name)
        self.model = DPTForDepthEstimation.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        # Vehicle masking parameters
        self.vehicle_mask_bottom = vehicle_mask_bottom
        self.vehicle_mask_sides = vehicle_mask_sides
        
        print("✓ Model loaded successfully")
    
    def create_vehicle_mask(self, shape: Tuple[int, int]) -> np.ndarray:
        """
        Create binary mask for vehicle body region.
        
        Args:
            shape: (height, width)
            
        Returns:
            Binary mask (H, W): 1 = valid region, 0 = vehicle body
        """
        h, w = shape
        mask = np.ones((h, w), dtype=np.uint8)
        
        # Mask bottom portion
        if self.vehicle_mask_bottom > 0:
            bottom_start = int(h * (1 - self.vehicle_mask_bottom))
            mask[bottom_start:, :] = 0
        
        # Mask sides (optional)
        if self.vehicle_mask_sides > 0:
            side_width = int(w * self.vehicle_mask_sides)
            mask[:, :side_width] = 0
            mask[:, -side_width:] = 0
        
        return mask
    
    def estimate_depth(self, image: np.ndarray) -> np.ndarray:
        """Estimate depth map."""
        pil_image = Image.fromarray(image)
        
        inputs = self.processor(images=pil_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            predicted_depth = outputs.predicted_depth
        
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=image.shape[:2],
            mode="bicubic",
            align_corners=False,
        )
        
        depth = prediction.squeeze().cpu().numpy()
        depth = (depth - depth.min()) / (depth.max() - depth.min())
        
        return depth
    
    def depth_to_segmentation(
        self,
        depth: np.ndarray,
        image: np.ndarray,
        vehicle_mask: np.ndarray,
        floor_threshold: float = 0.5,
        gradient_threshold: float = 0.12
    ) -> Tuple[np.ndarray, Dict]:
        """
        Convert depth map to segmentation with vehicle masking.
        
        Key difference from original:
        - Only processes pixels where vehicle_mask == 1
        - Vehicle body region is set to Background (0)
        """
        h, w = depth.shape
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # Compute gradient
        grad_y = np.abs(cv2.Sobel(depth, cv2.CV_64F, 0, 1, ksize=3))
        grad_x = np.abs(cv2.Sobel(depth, cv2.CV_64F, 1, 0, ksize=3))
        gradient = np.sqrt(grad_x**2 + grad_y**2)
        gradient = (gradient - gradient.min()) / (gradient.max() - gradient.min() + 1e-8)
        
        # Height weight - but adjusted for masked region
        # Only consider the valid region (above vehicle body)
        valid_height = int(h * (1 - self.vehicle_mask_bottom))
        height_weight = np.zeros((h, w))
        if valid_height > 0:
            # Normalize height only for valid region
            for i in range(valid_height):
                height_weight[i, :] = i / valid_height  # 0 at top, 1 at bottom of valid region
        
        # Apply vehicle mask to all features
        depth_masked = depth * vehicle_mask
        gradient_masked = gradient * vehicle_mask
        height_weight_masked = height_weight * vehicle_mask
        
        # Floor detection (more aggressive since vehicle body is excluded)
        floor_score = (
            (depth_masked > floor_threshold) * 0.4 +
            (gradient_masked < gradient_threshold) * 0.4 +
            (height_weight_masked > 0.2) * 0.2  # Lower region in valid area
        )
        
        # Obstacle detection
        obstacle_score = (
            (gradient_masked > gradient_threshold) * 0.5 +
            ((depth_masked < floor_threshold) * (height_weight_masked < 0.5)) * 0.3 +
            (depth_masked < 0.3) * 0.2
        )
        
        # Apply thresholds (only to valid regions)
        mask[floor_score > 0.5] = 1      # Road
        mask[obstacle_score > 0.4] = 2   # Obstacle
        
        # Ensure vehicle body region is Background (0)
        mask[vehicle_mask == 0] = 0
        
        # Post-processing
        mask = self._morphological_cleanup(mask, vehicle_mask)
        
        # Statistics (only count valid region)
        valid_pixels = np.sum(vehicle_mask)
        
        stats = {
            'total_pixels': mask.size,
            'valid_pixels': int(valid_pixels),
            'vehicle_body_pixels': int(mask.size - valid_pixels),
            'background': int(np.sum((mask == 0) & (vehicle_mask == 1))),  # Background in valid region
            'road': int(np.sum(mask == 1)),
            'obstacle': int(np.sum(mask == 2)),
        }
        
        # Percentages relative to VALID region
        if valid_pixels > 0:
            stats['background_pct'] = stats['background'] / valid_pixels * 100
            stats['road_pct'] = stats['road'] / valid_pixels * 100
            stats['obstacle_pct'] = stats['obstacle'] / valid_pixels * 100
        else:
            stats['background_pct'] = stats['road_pct'] = stats['obstacle_pct'] = 0
        
        return mask, stats
    
    def _morphological_cleanup(self, mask: np.ndarray, vehicle_mask: np.ndarray) -> np.ndarray:
        """Clean up mask."""
        kernel = np.ones((5, 5), np.uint8)
        
        for class_id in [1, 2]:
            class_mask = ((mask == class_id) & (vehicle_mask == 1)).astype(np.uint8)
            class_mask = cv2.morphologyEx(class_mask, cv2.MORPH_OPEN, kernel, iterations=1)
            class_mask = cv2.morphologyEx(class_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
            mask[class_mask > 0] = class_id
        
        return mask
    
    def annotate_image(self, image_path: Path) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Annotate image with vehicle masking."""
        print(f"Processing: {image_path.name}")
        
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Create vehicle mask
        vehicle_mask = self.create_vehicle_mask(image.shape[:2])
        
        print(f"  Vehicle body masked: {(1-vehicle_mask.mean())*100:.1f}% of image")
        
        print("  Estimating depth...")
        depth = self.estimate_depth(image)
        
        print("  Converting to segmentation (with vehicle masking)...")
        mask, stats = self.depth_to_segmentation(depth, image, vehicle_mask)
        
        print(f"  Valid region distribution:")
        print(f"    Background: {stats['background_pct']:.1f}%")
        print(f"    Road:       {stats['road_pct']:.1f}%")
        print(f"    Obstacle:   {stats['obstacle_pct']:.1f}%")
        
        return image, mask, stats
    
    def create_visualization(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        alpha: float = 0.5,
        show_vehicle_mask: bool = True
    ) -> np.ndarray:
        """
        Create visualization with optional vehicle mask overlay.
        
        Args:
            show_vehicle_mask: If True, show vehicle body region in dark gray
        """
        colored_mask = np.zeros_like(image)
        
        for class_id, color in CLASS_COLORS.items():
            if class_id == 3 and not show_vehicle_mask:
                continue
            colored_mask[mask == class_id] = color
        
        # Highlight vehicle body region
        if show_vehicle_mask:
            h, w = mask.shape
            vehicle_start = int(h * (1 - self.vehicle_mask_bottom))
            colored_mask[vehicle_start:, :] = CLASS_COLORS[3]  # Dark gray
        
        vis = cv2.addWeighted(image, 1 - alpha, colored_mask, alpha, 0)
        
        # Draw line to show vehicle mask boundary
        if show_vehicle_mask and self.vehicle_mask_bottom > 0:
            h, w = mask.shape
            vehicle_start = int(h * (1 - self.vehicle_mask_bottom))
            cv2.line(vis, (0, vehicle_start), (w, vehicle_start), (255, 255, 0), 2)  # Yellow line
        
        return vis


def process_images(
    annotator: MaskedDepthAnnotator,
    input_dir: Path,
    output_dir: Path,
    visualize: bool = False
) -> Dict:
    """Process all images."""
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
    
    stats = {
        'total': len(image_files),
        'processed': 0,
        'failed': 0
    }
    
    for idx, image_path in enumerate(image_files):
        try:
            image, mask, img_stats = annotator.annotate_image(image_path)
            
            mask_filename = image_path.stem + '_mask.png'
            mask_path = masks_dir / mask_filename
            cv2.imwrite(str(mask_path), mask)
            
            if visualize:
                vis = annotator.create_visualization(image, mask, show_vehicle_mask=True)
                vis_filename = image_path.stem + '_vis.jpg'
                vis_path = vis_dir / vis_filename
                cv2.imwrite(str(vis_path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
            
            stats['processed'] += 1
            
        except Exception as e:
            print(f"Error: {e}")
            stats['failed'] += 1
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description='Depth-based annotation with vehicle body masking'
    )
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--model', type=str, default='Intel/dpt-large')
    parser.add_argument('--vehicle-mask-bottom', type=float, default=0.25,
                       help='Fraction of image height to mask from bottom (default: 0.25 = 25%%)')
    parser.add_argument('--vehicle-mask-sides', type=float, default=0.0,
                       help='Fraction of image width to mask from each side (default: 0.0)')
    parser.add_argument('--visualize', action='store_true')
    
    args = parser.parse_args()
    
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    
    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        return 1
    
    print("=" * 60)
    print("Depth-Based Auto-Annotation with Vehicle Masking")
    print("=" * 60)
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Model: {args.model}")
    print(f"Vehicle mask: bottom {args.vehicle_mask_bottom*100:.0f}%")
    print("=" * 60)
    print()
    
    annotator = MaskedDepthAnnotator(
        args.model,
        vehicle_mask_bottom=args.vehicle_mask_bottom,
        vehicle_mask_sides=args.vehicle_mask_sides
    )
    
    print()
    stats = process_images(annotator, input_dir, output_dir, args.visualize)
    
    print()
    print("=" * 60)
    print("Masked Annotation Complete!")
    print("=" * 60)
    print(f"Total: {stats['total']}")
    print(f"Processed: {stats['processed']}")
    print(f"Failed: {stats['failed']}")
    print()
    print(f"Masks: {output_dir / 'masks'}")
    if args.visualize:
        print(f"Visualizations: {output_dir / 'visualizations'}")
        print("(Yellow line shows vehicle mask boundary)")
    print("=" * 60)
    
    return 0


if __name__ == '__main__':
    exit(main())
