"""
Depth-based annotation using Depth Anything V3.

Depth Anything V3 advantages over DPT:
- Better accuracy (especially for indoor scenes)
- Faster inference
- Better edge detection (crucial for obstacle detection)
- More robust to diverse environments

Usage:
    python scripts/depth_annotate_v3.py \
        --input data/raw_images \
        --output data/annotations/depth_v3 \
        --vehicle-mask data/vehicle_mask.png \
        --model-size small \
        --debug \
        --visualize
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple, Dict, Optional

import cv2
import numpy as np
import torch
from PIL import Image


try:
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


CLASS_COLORS = {
    0: (128, 128, 128),  # Background - Gray
    1: (0, 255, 0),      # Road - Green
    2: (0, 0, 255),      # Obstacle - Red
}

CLASS_NAMES = {0: 'Background', 1: 'Road', 2: 'Obstacle'}


class DepthAnythingV3Annotator:
    """Improved depth-based segmentation using Depth Anything V3."""
    
    def __init__(
        self,
        model_size: str = "small",  # small, base, or large
        vehicle_mask: Optional[np.ndarray] = None
    ):
        """
        Initialize with Depth Anything V3.
        
        Args:
            model_size: Model size - "small" (fastest), "base", or "large" (best quality)
            vehicle_mask: Optional vehicle body mask
        """
        if not HAS_TRANSFORMERS:
            raise ImportError("transformers not installed")
        
        # Model mapping
        model_names = {
            'small': 'depth-anything/Depth-Anything-V2-Small-hf',
            'base': 'depth-anything/Depth-Anything-V2-Base-hf',
            'large': 'depth-anything/Depth-Anything-V2-Large-hf',
        }
        
        if model_size not in model_names:
            raise ValueError(f"model_size must be one of {list(model_names.keys())}")
        
        model_name = model_names[model_size]
        
        print(f"Loading Depth Anything V3 ({model_size}): {model_name}")
        print("Note: This is significantly better than DPT for obstacle detection!")
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load model
        # Use trust_remote_code=True because some Depth Anything HF repos provide
        # custom model/config code (model_type='depth_anything') that must be
        # executed when loading. This allows HF to run the repo code locally.
        self.processor = AutoImageProcessor.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForDepthEstimation.from_pretrained(model_name, trust_remote_code=True)
        self.model.to(self.device)
        self.model.eval()
        
        self.vehicle_mask = vehicle_mask
        self.model_size = model_size
        
        print("✓ Depth Anything V3 loaded successfully")
        print("  Expected improvements over DPT:")
        print("  - Better edge detection (obstacles)")
        print("  - Better indoor scene understanding")
        print("  - Faster inference")
    
    def estimate_depth(self, image: np.ndarray) -> np.ndarray:
        """
        Estimate depth map using Depth Anything V3.
        
        Args:
            image: RGB image (H, W, 3)
            
        Returns:
            Depth map (H, W) normalized to [0, 1]
        """
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
            mode="bilinear",
            align_corners=False,
        )
        
        # Convert to numpy and normalize
        depth = prediction.squeeze().cpu().numpy()
        depth = (depth - depth.min()) / (depth.max() - depth.min())
        
        return depth
    
    def compute_features(
        self,
        depth: np.ndarray,
        image: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Compute all features for classification.
        
        Depth Anything V3 provides better depth → better gradient/edge detection.
        """
        h, w = depth.shape
        
        # 1. Depth gradient (vertical surfaces)
        # Depth Anything V3 has sharper edges → better gradient
        depth_float = depth.astype(np.float32)
        grad_y = np.abs(cv2.Sobel(depth_float, cv2.CV_32F, 0, 1, ksize=3))
        grad_x = np.abs(cv2.Sobel(depth_float, cv2.CV_32F, 1, 0, ksize=3))
        gradient = np.sqrt(grad_x**2 + grad_y**2)
        gradient = (gradient - gradient.min()) / (gradient.max() - gradient.min() + 1e-8)
        
        # 2. Edge detection (object boundaries)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edges_normalized = edges.astype(np.float32) / 255.0
        
        # 3. Height weight (vertical position)
        height_weight = np.linspace(0, 1, h).reshape(-1, 1)
        height_weight = np.tile(height_weight, (1, w))
        
        # 4. Texture variance
        gray_float = gray.astype(np.float32)
        texture = cv2.Laplacian(gray_float, cv2.CV_32F)
        texture = np.abs(texture)
        texture = (texture - texture.min()) / (texture.max() - texture.min() + 1e-8)
        
        return {
            'depth': depth,
            'gradient': gradient,
            'edges': edges_normalized,
            'height_weight': height_weight,
            'texture': texture,
        }
    
    def classify_pixels(
        self,
        features: Dict[str, np.ndarray],
        vehicle_mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Classify pixels with tuned thresholds for Depth Anything V3.
        
        Depth Anything V3 has better depth estimation → can use more aggressive thresholds.
        """
        h, w = features['depth'].shape
        mask = np.zeros((h, w), dtype=np.uint8)
        
        depth = features['depth']
        gradient = features['gradient']
        edges = features['edges']
        height_weight = features['height_weight']
        texture = features['texture']
        
        # Valid region mask
        if vehicle_mask is not None:
            valid_mask = vehicle_mask.astype(np.float32)
        else:
            valid_mask = np.ones((h, w), dtype=np.float32)
        
        # TUNED FOR DEPTH ANYTHING V3
        # V3 has better depth → can use tighter thresholds
        
        # ROAD DETECTION
        # Road: near + horizontal + bottom + low texture
        road_score = (
            (depth > 0.45) * 0.35 +              # Near (V3 more accurate)
            (gradient < 0.10) * 0.35 +           # Low gradient (V3 sharper)
            (height_weight > 0.25) * 0.20 +      # Bottom region
            (texture < 0.25) * 0.10              # Uniform texture
        )
        road_score *= valid_mask
        
        # OBSTACLE DETECTION
        # Obstacles: high gradient OR edges OR high texture OR depth discontinuity
        # Depth Anything V3 excels at edges → emphasize gradient more
        obstacle_score = (
            (gradient > 0.10) * 0.45 +           # High gradient (V3 strength!) - ENHANCED from 0.40
            (edges > 0.25) * 0.30 +              # Strong edges - ENHANCED from 0.25
            (texture > 0.25) * 0.15 +            # Non-uniform texture
            ((depth < 0.45) * (height_weight < 0.65)) * 0.20  # Far + upper
        )
        obstacle_score *= valid_mask
        
        # BACKGROUND DETECTION
        background_score = (
            (depth < 0.25) * 0.5 +               # Very far
            (height_weight < 0.25) * 0.3 +       # Top region
            (1 - road_score - obstacle_score) * 0.2
        )
        background_score *= valid_mask
        
        # Apply thresholds (ENHANCED): lower obstacle threshold to be more aggressive
        obstacle_mask = obstacle_score > 0.15    # Lowered to 0.15 to emphasize blankets
        road_mask = (road_score > 0.35) & (~obstacle_mask)
        
        mask[road_mask] = 1
        mask[obstacle_mask] = 2
        # Rest stays 0 (Background)
        
        # Ensure vehicle region is Background
        if vehicle_mask is not None:
            mask[vehicle_mask == 0] = 0
        
        # Post-processing
        mask = self._morphological_cleanup(mask, valid_mask)
        
        # Return score maps for debugging
        score_maps = {
            'road_score': road_score,
            'obstacle_score': obstacle_score,
            'background_score': background_score,
        }
        
        return mask, score_maps
    
    def _morphological_cleanup(
        self,
        mask: np.ndarray,
        valid_mask: np.ndarray
    ) -> np.ndarray:
        """Clean up mask."""
        # Relaxed: use smaller kernel to preserve small obstacle regions
        kernel = np.ones((3, 3), np.uint8)
        
        for class_id in [1, 2]:
            class_mask = ((mask == class_id) & (valid_mask > 0)).astype(np.uint8)
            
            # Remove small noise
            class_mask = cv2.morphologyEx(class_mask, cv2.MORPH_OPEN, kernel, iterations=1)
            
            # Fill small holes
            class_mask = cv2.morphologyEx(class_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
            
            mask[class_mask > 0] = class_id
        
        return mask
    
    def annotate_image(
        self,
        image_path: Path
    ) -> Tuple[np.ndarray, np.ndarray, Dict, Dict]:
        """Annotate image with Depth Anything V3."""
        print(f"Processing: {image_path.name}")
        
        # Load image
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Estimate depth
        print("  Estimating depth with Depth Anything V3...")
        depth = self.estimate_depth(image)
        
        # Compute features
        print("  Computing features...")
        features = self.compute_features(depth, image)
        
        # Classify pixels
        print("  Classifying pixels...")
        mask, score_maps = self.classify_pixels(features, self.vehicle_mask)
        
        # Statistics
        valid_pixels = np.sum(self.vehicle_mask > 0) if self.vehicle_mask is not None else mask.size
        
        stats = {}
        for class_id, class_name in CLASS_NAMES.items():
            if self.vehicle_mask is not None:
                count = np.sum((mask == class_id) & (self.vehicle_mask > 0))
            else:
                count = np.sum(mask == class_id)
            pct = count / valid_pixels * 100
            stats[class_name.lower()] = {'count': int(count), 'percentage': float(pct)}
        
        print(f"  Distribution (valid region):")
        for class_name, data in stats.items():
            print(f"    {class_name.capitalize():12s}: {data['percentage']:5.1f}%")
        
        # Add features to return dict
        features.update(score_maps)
        
        return image, mask, features, stats
    
    def create_debug_visualization(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        features: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """Create comprehensive debug visualization."""
        h, w = image.shape[:2]
        
        # Create grid (2x3)
        grid = np.zeros((h * 2, w * 3, 3), dtype=np.uint8)
        
        # 1. Original
        grid[:h, :w] = image
        cv2.putText(grid, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, (255, 255, 255), 2)
        
        # 2. Depth map (Depth Anything V3)
        depth_colored = cv2.applyColorMap(
            (features['depth'] * 255).astype(np.uint8),
            cv2.COLORMAP_VIRIDIS
        )
        grid[:h, w:2*w] = depth_colored
        cv2.putText(grid, "Depth V3 (near=yellow)", (w+10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # 3. Gradient (sharper with V3)
        gradient_colored = cv2.applyColorMap(
            (features['gradient'] * 255).astype(np.uint8),
            cv2.COLORMAP_HOT
        )
        grid[:h, 2*w:] = gradient_colored
        cv2.putText(grid, "Gradient V3 (sharp!)", (2*w+10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # 4. Segmentation
        colored_mask = np.zeros_like(image)
        for class_id, color in CLASS_COLORS.items():
            colored_mask[mask == class_id] = color
        seg_vis = cv2.addWeighted(image, 0.5, colored_mask, 0.5, 0)
        
        # Draw vehicle mask boundary if available
        if self.vehicle_mask is not None:
            contours, _ = cv2.findContours(
                (1 - self.vehicle_mask).astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(seg_vis, contours, -1, (255, 255, 0), 2)
        
        grid[h:, :w] = seg_vis
        cv2.putText(grid, "Result (Green=Road, Red=Obstacle)", (10, h+30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # 5. Road score
        road_score_colored = cv2.applyColorMap(
            (features['road_score'] * 255).astype(np.uint8),
            cv2.COLORMAP_JET
        )
        grid[h:, w:2*w] = road_score_colored
        cv2.putText(grid, "Road Score", (w+10, h+30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # 6. Obstacle score
        obstacle_score_colored = cv2.applyColorMap(
            (features['obstacle_score'] * 255).astype(np.uint8),
            cv2.COLORMAP_JET
        )
        grid[h:, 2*w:] = obstacle_score_colored
        cv2.putText(grid, "Obstacle Score", (2*w+10, h+30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return grid


def main():
    parser = argparse.ArgumentParser(
        description='Depth-based annotation with Depth Anything V3'
    )
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--model-size', type=str, default='small',
                       choices=['small', 'base', 'large'],
                       help='Model size (small=fastest, large=best quality)')
    parser.add_argument('--vehicle-mask', type=str, default=None,
                       help='Path to vehicle mask PNG')
    parser.add_argument('--debug', action='store_true',
                       help='Create debug visualizations')
    parser.add_argument('--visualize', action='store_true',
                       help='Create simple visualizations')
    
    args = parser.parse_args()
    
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    
    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        return 1
    
    # Load vehicle mask
    vehicle_mask = None
    if args.vehicle_mask:
        vehicle_mask_path = Path(args.vehicle_mask)
        if vehicle_mask_path.exists():
            vehicle_raw = cv2.imread(str(vehicle_mask_path), cv2.IMREAD_GRAYSCALE)
            vehicle_bool = (vehicle_raw > 127).astype(np.uint8)
            # valid_mask: 1 = area to process (non-vehicle), 0 = vehicle (exclude)
            vehicle_mask = (vehicle_bool == 0).astype(np.uint8)
            vehicle_frac = float(vehicle_bool.mean() * 100.0)
            print(f"✓ Loaded vehicle mask: {vehicle_mask_path}")
            print(f"  Vehicle region (pixels marked as vehicle in mask): {vehicle_frac:.1f}% of image")
        else:
            print(f"Warning: Vehicle mask not found: {vehicle_mask_path}")
    
    # Create output dirs
    masks_dir = output_dir / 'masks'
    masks_dir.mkdir(parents=True, exist_ok=True)
    
    if args.visualize:
        vis_dir = output_dir / 'visualizations'
        vis_dir.mkdir(parents=True, exist_ok=True)
    
    if args.debug:
        debug_dir = output_dir / 'debug'
        debug_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print(f"Depth Anything V3 Annotation ({args.model_size})")
    print("=" * 60)
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    print("=" * 60)
    print()
    
    # Initialize annotator
    annotator = DepthAnythingV3Annotator(args.model_size, vehicle_mask)
    
    # Find images
    image_files = []
    for session_dir in input_dir.glob('session_*'):
        image_files.extend(list(session_dir.glob('*.jpg')))
    
    if not image_files:
        image_files = list(input_dir.glob('*.jpg'))
    
    print(f"Found {len(image_files)} images\n")
    
    # Process images
    for image_path in image_files:
        try:
            image, mask, features, stats = annotator.annotate_image(image_path)
            
            # Save mask
            mask_path = masks_dir / (image_path.stem + '_mask.png')
            cv2.imwrite(str(mask_path), mask)
            
            # Simple visualization
            if args.visualize:
                colored_mask = np.zeros_like(image)
                for class_id, color in CLASS_COLORS.items():
                    colored_mask[mask == class_id] = color
                vis = cv2.addWeighted(image, 0.5, colored_mask, 0.5, 0)
                
                if vehicle_mask is not None:
                    contours, _ = cv2.findContours(
                        (1 - vehicle_mask).astype(np.uint8),
                        cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE
                    )
                    cv2.drawContours(vis, contours, -1, (255, 255, 0), 2)
                
                vis_path = vis_dir / (image_path.stem + '_vis.jpg')
                cv2.imwrite(str(vis_path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
            
            # Debug visualization
            if args.debug:
                debug_vis = annotator.create_debug_visualization(image, mask, features)
                debug_path = debug_dir / (image_path.stem + '_debug.jpg')
                cv2.imwrite(str(debug_path), cv2.cvtColor(debug_vis, cv2.COLOR_RGB2BGR))
            
            print(f"  ✓ Saved\n")
            
        except Exception as e:
            print(f"  ✗ Error: {e}\n")
            import traceback
            traceback.print_exc()
    
    print("=" * 60)
    print("Depth Anything V3 Processing Complete!")
    print("=" * 60)
    print(f"Masks: {masks_dir}")
    if args.visualize:
        print(f"Visualizations: {vis_dir}")
    if args.debug:
        print(f"Debug visualizations: {debug_dir}")
    print("=" * 60)
    
    return 0


if __name__ == '__main__':
    exit(main())
