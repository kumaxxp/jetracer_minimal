"""
Generate vehicle mask from static region detection across multiple images.

This script analyzes multiple images to find regions that don't change
(the vehicle body) and creates a binary mask for those regions.

Usage:
    python scripts/generate_vehicle_mask.py \
        --input data/raw_images \
        --output data/vehicle_mask.png \
        --num-samples 5 \
        --visualize
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np


class StaticRegionDetector:
    """Detect static regions (vehicle body) from multiple images."""
    
    def __init__(
        self,
        difference_threshold: float = 20.0,
        static_ratio_threshold: float = 0.8,
        bottom_region_ratio: float = 0.5,
        morphology_kernel_size: int = 15
    ):
        """
        Initialize detector.
        
        Args:
            difference_threshold: Pixel difference threshold (0-255)
            static_ratio_threshold: Ratio of pairs that must agree (0-1)
            bottom_region_ratio: Only consider bottom X of image (0-1)
            morphology_kernel_size: Kernel size for cleanup
        """
        self.difference_threshold = difference_threshold
        self.static_ratio_threshold = static_ratio_threshold
        self.bottom_region_ratio = bottom_region_ratio
        self.morphology_kernel_size = morphology_kernel_size
        
        print("Static Region Detector initialized:")
        print(f"  Difference threshold: {difference_threshold}")
        print(f"  Static ratio threshold: {static_ratio_threshold}")
        print(f"  Bottom region ratio: {bottom_region_ratio}")
    
    def compute_frame_difference(
        self,
        img1: np.ndarray,
        img2: np.ndarray
    ) -> np.ndarray:
        """
        Compute pixel-wise difference between two frames.
        
        Returns:
            Binary mask: 1 = static, 0 = changed
        """
        # Convert to grayscale
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        # Compute absolute difference
        diff = cv2.absdiff(gray1, gray2)
        
        # Threshold: 1 = static (small difference)
        static_mask = (diff < self.difference_threshold).astype(np.uint8)
        
        return static_mask
    
    def detect_static_regions(
        self,
        images: List[np.ndarray]
    ) -> Tuple[np.ndarray, dict]:
        """
        Detect static regions across multiple images.
        
        Args:
            images: List of images (same size)
            
        Returns:
            Tuple of (mask, statistics)
        """
        if len(images) < 2:
            raise ValueError("Need at least 2 images")
        
        h, w = images[0].shape[:2]
        
        # Only process bottom region
        bottom_start = int(h * (1 - self.bottom_region_ratio))
        
        print(f"\nProcessing {len(images)} images...")
        print(f"Analyzing bottom {self.bottom_region_ratio*100:.0f}% "
              f"(rows {bottom_start}-{h})")
        
        # Accumulator for static pixels
        static_count = np.zeros((h, w), dtype=np.float32)
        total_comparisons = 0
        
        # Compare all pairs
        for i in range(len(images)):
            for j in range(i + 1, len(images)):
                # Compute difference
                static_mask = self.compute_frame_difference(images[i], images[j])
                
                # Only accumulate for bottom region
                static_mask[:bottom_start, :] = 0
                
                static_count += static_mask
                total_comparisons += 1
        
        print(f"Total comparisons: {total_comparisons}")
        
        # Normalize: ratio of pairs where pixel was static
        static_ratio = static_count / total_comparisons
        
        # Threshold: pixels that are static in most comparisons
        vehicle_mask = (static_ratio > self.static_ratio_threshold).astype(np.uint8)
        
        # Morphological operations to clean up
        kernel = np.ones(
            (self.morphology_kernel_size, self.morphology_kernel_size),
            np.uint8
        )
        
        # Close small holes
        vehicle_mask = cv2.morphologyEx(
            vehicle_mask, cv2.MORPH_CLOSE, kernel, iterations=2
        )
        
        # Remove small regions
        vehicle_mask = cv2.morphologyEx(
            vehicle_mask, cv2.MORPH_OPEN, kernel, iterations=1
        )
        
        # Keep only the largest connected component in bottom region
        vehicle_mask = self._keep_largest_component_bottom(vehicle_mask, bottom_start)
        
        # Statistics
        total_pixels = h * w
        bottom_pixels = (h - bottom_start) * w
        vehicle_pixels = np.sum(vehicle_mask)
        
        stats = {
            'total_pixels': total_pixels,
            'bottom_region_pixels': bottom_pixels,
            'vehicle_pixels': int(vehicle_pixels),
            'vehicle_ratio_total': vehicle_pixels / total_pixels,
            'vehicle_ratio_bottom': vehicle_pixels / bottom_pixels,
            'num_comparisons': total_comparisons,
        }
        
        return vehicle_mask, stats
    
    def _keep_largest_component_bottom(
        self,
        mask: np.ndarray,
        bottom_start: int
    ) -> np.ndarray:
        """Keep only largest connected component in bottom region."""
        h, w = mask.shape
        result = np.zeros_like(mask)
        
        # Find connected components in bottom region only
        bottom_mask = mask[bottom_start:, :].copy()
        
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            bottom_mask, connectivity=8
        )
        
        if num_labels <= 1:  # Only background
            return result
        
        # Find largest component (excluding background at index 0)
        areas = stats[1:, cv2.CC_STAT_AREA]
        largest_idx = np.argmax(areas) + 1
        
        # Keep only largest component
        bottom_result = (labels == largest_idx).astype(np.uint8)
        result[bottom_start:, :] = bottom_result
        
        return result
    
    def create_visualization(
        self,
        sample_image: np.ndarray,
        vehicle_mask: np.ndarray,
        static_ratio: np.ndarray = None
    ) -> np.ndarray:
        """Create debug visualization."""
        vis = sample_image.copy()
        
        # Overlay vehicle mask in red
        red_mask = np.zeros_like(vis)
        red_mask[vehicle_mask > 0] = [0, 0, 255]
        vis = cv2.addWeighted(vis, 0.7, red_mask, 0.3, 0)
        
        # Draw boundary
        contours, _ = cv2.findContours(
            vehicle_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(vis, contours, -1, (0, 255, 255), 2)
        
        # Add text
        h, w = vehicle_mask.shape
        vehicle_pixels = np.sum(vehicle_mask)
        ratio = vehicle_pixels / (h * w) * 100
        
        cv2.putText(
            vis,
            f"Vehicle Region: {ratio:.1f}% of image",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 255),
            2
        )
        
        return vis


def load_random_images(
    input_dir: Path,
    num_samples: int
) -> List[np.ndarray]:
    """Load random sample of images."""
    # Find all images
    image_files = []
    for session_dir in input_dir.glob('session_*'):
        image_files.extend(list(session_dir.glob('*.jpg')))
    
    if not image_files:
        image_files = list(input_dir.glob('*.jpg'))
    
    if len(image_files) < 2:
        raise ValueError(f"Need at least 2 images, found {len(image_files)}")
    
    # Sample randomly
    num_samples = min(num_samples, len(image_files))
    sampled_files = random.sample(image_files, num_samples)
    
    print(f"Loading {num_samples} images from {len(image_files)} total...")
    
    # Load images
    images = []
    for img_path in sampled_files:
        img = cv2.imread(str(img_path))
        if img is not None:
            images.append(img)
            print(f"  ✓ {img_path.name}")
    
    if len(images) < 2:
        raise ValueError("Failed to load at least 2 images")
    
    return images


def main():
    parser = argparse.ArgumentParser(
        description='Generate vehicle mask from static region detection'
    )
    parser.add_argument('--input', type=str, required=True,
                       help='Input directory with images')
    parser.add_argument('--output', type=str, required=True,
                       help='Output mask file (PNG)')
    parser.add_argument('--num-samples', type=int, default=5,
                       help='Number of images to sample')
    parser.add_argument('--difference-threshold', type=float, default=20.0,
                       help='Pixel difference threshold (0-255)')
    parser.add_argument('--static-ratio', type=float, default=0.8,
                       help='Static ratio threshold (0-1)')
    parser.add_argument('--bottom-region', type=float, default=0.5,
                       help='Bottom region ratio (0-1)')
    parser.add_argument('--visualize', action='store_true',
                       help='Create visualization')
    
    args = parser.parse_args()
    
    input_dir = Path(args.input)
    output_path = Path(args.output)
    
    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        return 1
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Vehicle Mask Generation from Static Region Detection")
    print("=" * 60)
    print(f"Input: {input_dir}")
    print(f"Output: {output_path}")
    print(f"Samples: {args.num_samples}")
    print("=" * 60)
    
    # Load images
    images = load_random_images(input_dir, args.num_samples)
    
    # Initialize detector
    detector = StaticRegionDetector(
        difference_threshold=args.difference_threshold,
        static_ratio_threshold=args.static_ratio,
        bottom_region_ratio=args.bottom_region
    )
    
    # Detect static regions
    vehicle_mask, stats = detector.detect_static_regions(images)
    
    # Save mask
    cv2.imwrite(str(output_path), vehicle_mask * 255)
    print(f"\n✓ Vehicle mask saved: {output_path}")
    
    # Print statistics
    print("\n" + "=" * 60)
    print("Vehicle Mask Statistics:")
    print("=" * 60)
    print(f"Total pixels: {stats['total_pixels']:,}")
    print(f"Bottom region pixels: {stats['bottom_region_pixels']:,}")
    print(f"Vehicle pixels: {stats['vehicle_pixels']:,}")
    print(f"Vehicle ratio (total): {stats['vehicle_ratio_total']*100:.1f}%")
    print(f"Vehicle ratio (bottom): {stats['vehicle_ratio_bottom']*100:.1f}%")
    print("=" * 60)
    
    # Create visualization
    if args.visualize:
        vis = detector.create_visualization(images[0], vehicle_mask)
        vis_path = output_path.parent / (output_path.stem + '_visualization.jpg')
        cv2.imwrite(str(vis_path), vis)
        print(f"✓ Visualization saved: {vis_path}")
    
    return 0


if __name__ == '__main__':
    exit(main())
