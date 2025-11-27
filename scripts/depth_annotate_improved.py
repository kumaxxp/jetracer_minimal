"""
Improved depth-based annotation with comprehensive debug visualization.

Features:
- Automatic vehicle mask from static region detection
- Stricter obstacle detection using depth + gradient + edges
- Comprehensive debug visualization showing classification reasoning

Usage:
    python scripts/depth_annotate_improved.py \
        --input data/raw_images \
        --output data/annotations/depth_improved \
        --vehicle-mask data/vehicle_mask.png \
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
    from transformers import DPTImageProcessor, DPTForDepthEstimation
    HAS_DPT = True
except ImportError:
    HAS_DPT = False


CLASS_COLORS = {
    0: (128, 128, 128),  # Background - Gray
    1: (0, 255, 0),      # Road - Green
    2: (0, 0, 255),      # Obstacle - Red
}

CLASS_NAMES = {0: 'Background', 1: 'Road', 2: 'Obstacle'}


class ImprovedDepthAnnotator:
    """Improved depth-based segmentation with debug info."""
    
    def __init__(
        self,
        model_name: str = "Intel/dpt-large",
        vehicle_mask: Optional[np.ndarray] = None
    ):
        if not HAS_DPT:
            raise ImportError("transformers not installed")
        
        print(f"Loading depth model: {model_name}")
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self.processor = DPTImageProcessor.from_pretrained(model_name)
        self.model = DPTForDepthEstimation.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        self.vehicle_mask = vehicle_mask
        
        print("✓ Model loaded successfully")
    
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
    
    def compute_features(
        self,
        depth: np.ndarray,
        image: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Compute all features for classification.
        
        Returns:
            Dictionary of feature maps
        """
        h, w = depth.shape
        
        # 1. Depth gradient (vertical surfaces)
        grad_y = np.abs(cv2.Sobel(depth, cv2.CV_64F, 0, 1, ksize=3))
        grad_x = np.abs(cv2.Sobel(depth, cv2.CV_64F, 1, 0, ksize=3))
        gradient = np.sqrt(grad_x**2 + grad_y**2)
        gradient = (gradient - gradient.min()) / (gradient.max() - gradient.min() + 1e-8)
        
        # 2. Edge detection (object boundaries)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edges_normalized = edges.astype(np.float32) / 255.0
        
        # 3. Height weight (vertical position in image)
        # Top = 0, Bottom = 1
        height_weight = np.linspace(0, 1, h).reshape(-1, 1)
        height_weight = np.tile(height_weight, (1, w))
        
        # 4. Texture variance (differentiate floor from objects)
        gray_float = gray.astype(np.float32)
        texture = cv2.Laplacian(gray_float, cv2.CV_64F)
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
        Classify pixels with debug score maps.
        
        Returns:
            Tuple of (mask, score_maps)
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
        
        # IMPROVED ROAD DETECTION
        # Road: near + horizontal + bottom region + low texture variance
        road_score = (
            (depth > 0.4) * 0.3 +                    # Near regions
            (gradient < 0.12) * 0.3 +                # Low gradient (horizontal)
            (height_weight > 0.3) * 0.2 +            # Bottom region
            (texture < 0.3) * 0.2                    # Uniform texture (floor)
        )
        road_score *= valid_mask
        
        # IMPROVED OBSTACLE DETECTION
        # Obstacles: high gradient OR edges OR high texture variance
        obstacle_score = (
            (gradient > 0.12) * 0.35 +               # High gradient (vertical)
            (edges > 0.3) * 0.25 +                   # Strong edges
            (texture > 0.3) * 0.20 +                 # Non-uniform texture
            ((depth < 0.4) * (height_weight < 0.7)) * 0.20  # Far + upper
        )
        obstacle_score *= valid_mask
        
        # BACKGROUND DETECTION
        # Background: far regions + top of image
        background_score = (
            (depth < 0.3) * 0.5 +                    # Far regions
            (height_weight < 0.3) * 0.3 +            # Top region
            (1 - road_score - obstacle_score) * 0.2  # Not clearly road or obstacle
        )
        background_score *= valid_mask
        
        # Apply thresholds with priority: Obstacle > Road > Background
        obstacle_mask = obstacle_score > 0.45
        road_mask = (road_score > 0.45) & (~obstacle_mask)
        
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
        kernel = np.ones((5, 5), np.uint8)
        
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
        """
        Annotate image with debug info.
        
        Returns:
            Tuple of (image, mask, features, score_maps)
        """
        print(f"Processing: {image_path.name}")
        
        # Load image
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Estimate depth
        print("  Estimating depth...")
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
        """
        Create comprehensive debug visualization.
        
        Layout:
        [Original] [Depth]     [Gradient]
        [Segmentation] [Road Score] [Obstacle Score]
        """
        h, w = image.shape[:2]
        
        # Create grid
        grid = np.zeros((h * 2, w * 3, 3), dtype=np.uint8)
        
        # 1. Original image
        grid[:h, :w] = image
        cv2.putText(grid, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, (255, 255, 255), 2)
        
        # 2. Depth map
        depth_colored = cv2.applyColorMap(
            (features['depth'] * 255).astype(np.uint8),
            cv2.COLORMAP_VIRIDIS
        )
        grid[:h, w:2*w] = depth_colored
        cv2.putText(grid, "Depth (near=yellow, far=purple)", (w+10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # 3. Gradient map
        gradient_colored = cv2.applyColorMap(
            (features['gradient'] * 255).astype(np.uint8),
            cv2.COLORMAP_HOT
        )
        grid[:h, 2*w:] = gradient_colored
        cv2.putText(grid, "Gradient (high=red)", (2*w+10, 30),
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
        cv2.putText(grid, "Segmentation (Green=Road, Red=Obstacle)", (10, h+30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # 5. Road score
        road_score_colored = cv2.applyColorMap(
            (features['road_score'] * 255).astype(np.uint8),
            cv2.COLORMAP_JET
        )
        grid[h:, w:2*w] = road_score_colored
        cv2.putText(grid, "Road Score (high=red)", (w+10, h+30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # 6. Obstacle score
        obstacle_score_colored = cv2.applyColorMap(
            (features['obstacle_score'] * 255).astype(np.uint8),
            cv2.COLORMAP_JET
        )
        grid[h:, 2*w:] = obstacle_score_colored
        cv2.putText(grid, "Obstacle Score (high=red)", (2*w+10, h+30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return grid


def main():
    parser = argparse.ArgumentParser(
        description='Improved depth-based annotation with debug info'
    )
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--model', type=str, default='Intel/dpt-large')
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
            vehicle_mask = cv2.imread(str(vehicle_mask_path), cv2.IMREAD_GRAYSCALE)
            vehicle_mask = (vehicle_mask > 127).astype(np.uint8)
            print(f"✓ Loaded vehicle mask: {vehicle_mask_path}")
            print(f"  Vehicle region: {(1-vehicle_mask.mean())*100:.1f}% of image")
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
    print("Improved Depth-Based Annotation")
    print("=" * 60)
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Debug mode: {args.debug}")
    print("=" * 60)
    print()
    
    # Initialize annotator
    annotator = ImprovedDepthAnnotator(args.model, vehicle_mask)
    
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
            
            print(f"  ✓ Saved: {mask_path.name}\n")
            
        except Exception as e:
            print(f"  ✗ Error: {e}\n")
    
    print("=" * 60)
    print("Processing Complete!")
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
