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
import logging

# Configure logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


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

# ADE20K クラスのうち "rug/carpet/cushion/mat" 等は個別に保持して
# ユーザが後で通過可能/不可を判断できるようにする
PRESERVE_SYNONYMS = {
    'rug', 'carpet', 'carpeting', 'cushion', 'mat'
}

# Colors for visualization
CLASS_COLORS = {
    0: (128, 128, 128),  # Background - Gray
    1: (0, 255, 0),      # Road - Green
    2: (0, 0, 255),      # Obstacle - Red
    3: (64, 64, 64),     # Vehicle/対象外 - Dark Gray
}


class AutoAnnotator:
    """Automatic annotation using OneFormer."""
    
    def __init__(self, model_name: str = "shi-labs/oneformer_ade20k_swin_tiny"):
        """
        Initialize OneFormer model.
        
        Args:
            model_name: HuggingFace model name
        """
        logger.info(f"Loading model: {model_name}")
        logger.info("This may take a few minutes on first run...")
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Load processor and model
        self.processor = OneFormerProcessor.from_pretrained(model_name)
        self.model = OneFormerForUniversalSegmentation.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        # Get ADE20K class names from model config (tokenizer vocab is unrelated)
        config_id2label = self.model.config.id2label
        # Some configs store keys as strings, so normalize to ints
        self.id2label = {
            int(k) if isinstance(k, str) and k.isdigit() else k: v
            for k, v in config_id2label.items()
        }
        # Build synonym map from ADE20K_TO_JETRACER mapping keys.
        # Keys may contain multiple comma-separated synonyms like 'carpet, carpeting, rug'.
        self.synonym_map: Dict[str, int] = {}
        for key, cls in ADE20K_TO_JETRACER.items():
            parts = [p.strip().lower() for p in key.split(',') if p.strip()]
            for p in parts:
                self.synonym_map[p] = cls

        # Preserve certain visual classes (so user can later mark passable or not)
        # Assign unique labels for preserved ADE20K synonyms starting from 10
        self.preserve_map: Dict[str, int] = {}
        base_preserve_id = 10
        for syn in sorted(PRESERVE_SYNONYMS):
            if syn in self.synonym_map:
                self.preserve_map[syn] = base_preserve_id
                base_preserve_id += 1

        # Build per-instance class color map starting from base CLASS_COLORS
        self.class_colors = CLASS_COLORS.copy()
        # Generate distinct colors for preserved classes
        def gen_color(i, total):
            import colorsys
            h = i / max(total, 1)
            r, g, b = colorsys.hsv_to_rgb(h, 0.7, 0.9)
            return (int(r * 255), int(g * 255), int(b * 255))

        if self.preserve_map:
            total = len(self.preserve_map)
            for idx, (syn, lid) in enumerate(self.preserve_map.items()):
                self.class_colors[lid] = gen_color(idx, total)
        
        logger.info("✓ Model loaded successfully")
    
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

            # Get class name and normalize for mapping lookup
            class_name = (self.id2label.get(ade20k_id, '') or '').lower().strip()

            # Default
            jetracer_class = 0

            # 1) Preserve check (exact then fuzzy): if class matches a preserved synonym, assign preserved id
            if class_name in self.preserve_map:
                jetracer_class = self.preserve_map[class_name]
            else:
                # fuzzy match against preserve_map keys
                for syn, pid in self.preserve_map.items():
                    if syn in class_name or class_name in syn:
                        jetracer_class = pid
                        break

            # 2) If not preserved, map to JetRacer via synonym_map (exact then fuzzy)
            if jetracer_class == 0:
                if class_name in self.synonym_map:
                    jetracer_class = self.synonym_map[class_name]
                else:
                    for syn, cls in self.synonym_map.items():
                        if syn in class_name or class_name in syn:
                            jetracer_class = cls
                            break

            # Apply mapping
            jetracer_mask[ade20k_mask == ade20k_id] = jetracer_class
        
        return jetracer_mask
    
    def annotate_image(self, image_path: Path, vehicle_mask: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate annotation for a single image.
        
        Args:
            image_path: Path to input image
            vehicle_mask: Optional np.ndarray, 1=vehicle, 0=not vehicle
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

        # Apply vehicle mask: 車体領域は新クラス(3=対象外)に分類
        if vehicle_mask is not None:
            # 画像ごとにvehicle_maskをリサイズして適用
            if vehicle_mask.shape != jetracer_mask.shape:
                vehicle_mask = cv2.resize(vehicle_mask, (jetracer_mask.shape[1], jetracer_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
            vehicle_mask = (vehicle_mask > 0).astype(np.uint8)
            logger.debug('vehicle_mask > 0 count: %d', int(np.sum(vehicle_mask > 0)))
            jetracer_mask = jetracer_mask.copy()
            jetracer_mask[vehicle_mask.astype(bool)] = 3
            logger.debug('after mask assign, 3 count: %d', int(np.sum(jetracer_mask==3)))

        return image_np, jetracer_mask, ade20k_mask
    
    def create_visualization(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        vehicle_mask: np.ndarray = None,
        alpha: float = 0.5
    ) -> np.ndarray:
        """
        Create visualization by overlaying mask on image.
        
        Args:
            image: Original image (H, W, 3)
            mask: Segmentation mask (H, W)
            vehicle_mask: Optional, 1=vehicle, 0=not vehicle
            alpha: Overlay transparency
        Returns:
            Visualization image (H, W, 3)
        """
        # Create colored mask using per-instance colors (self.class_colors)
        colored_mask = np.zeros_like(image)
        for class_id, color in self.class_colors.items():
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
    # vehicle_maskの自動推定: 画像パスの親ディレクトリにある `session_*` を探す
    session_name = None
    vehicle_mask_path = None
    vehicle_mask = None
    if image_files:
        # find a parent directory that starts with 'session_' for the first image
        first_img = image_files[0]
        for p in first_img.parents:
            if p.name.startswith('session_'):
                session_name = p.name
                break
        if session_name:
            vehicle_mask_path = Path(f"data/vehicle_masks/{session_name}.png")
            if vehicle_mask_path.exists():
                vehicle_mask = cv2.imread(str(vehicle_mask_path), cv2.IMREAD_GRAYSCALE)
                # keep as 0/255 for now; will binarize per-image when applying
            else:
                print(f"[WARN] Vehicle mask not found: {vehicle_mask_path}")

    for idx, image_path in enumerate(image_files):
        try:
            print(f"Processing [{idx+1}/{len(image_files)}]: {image_path.name}")
            # 画像ごとにvehicle_maskをリサイズ
            vmask = None
            if vehicle_mask is not None:
                img0 = cv2.imread(str(image_path))
                h, w = img0.shape[:2]
                vmask = cv2.resize(vehicle_mask, (w, h), interpolation=cv2.INTER_NEAREST)
            # Annotate
            image, mask, ade20k_mask = annotator.annotate_image(image_path, vmask)
            # Save mask
            mask_filename = image_path.stem + '_mask.png'
            mask_path = masks_dir / mask_filename
            from PIL import Image as PILImage
            logger.debug('before save, unique: %s', np.array2string(np.unique(mask)))
            PILImage.fromarray(mask.astype(np.uint8)).save(str(mask_path))
            mask2 = np.array(PILImage.open(str(mask_path)))
            logger.debug('after save, unique: %s', np.array2string(np.unique(mask2)))
            # Create visualization
            if visualize:
                vis = annotator.create_visualization(image, mask, vmask)
                vis_filename = image_path.stem + '_vis.jpg'
                vis_path = vis_dir / vis_filename
                if vis.shape[2] == 3:
                    vis_bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
                else:
                    vis_bgr = vis
                cv2.imwrite(str(vis_path), vis_bgr)
            # Optionally save raw ADE20K mask + visualization if annotator has flag set
            if getattr(annotator, 'save_ade20k', False):
                try:
                    ade_filename = image_path.stem + '_ade20k.png'
                    ade_dir = output_dir / 'ade20k_masks'
                    ade_dir.mkdir(parents=True, exist_ok=True)
                    PILImage.fromarray(ade20k_mask.astype(np.uint8)).save(str(ade_dir / ade_filename))

                    # ADE20K visualization: deterministic color per ADE id
                    ade_vis = np.zeros_like(image)
                    def ade_color(cid: int):
                        import colorsys
                        h = (cid % 150) / 150.0
                        r, g, b = colorsys.hsv_to_rgb(h, 0.6, 0.9)
                        return (int(r*255), int(g*255), int(b*255))

                    for cid in np.unique(ade20k_mask):
                        ade_vis[ade20k_mask == cid] = ade_color(int(cid))

                    ade_vis_dir = output_dir / 'ade20k_visualizations'
                    ade_vis_dir.mkdir(parents=True, exist_ok=True)
                    ade_vis_bgr = cv2.cvtColor(ade_vis, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(str(ade_vis_dir / (image_path.stem + '_ade20k_vis.jpg')), ade_vis_bgr)
                except Exception:
                    logger.exception('Failed saving ADE20K outputs for %s', image_path.name)
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
    parser.add_argument(
        '--save-ade20k',
        action='store_true',
        help='Save raw ADE20K masks and ADE20K visualizations'
    )

    args = parser.parse_args()

    # Convert paths
    input_dir = Path(args.input)
    output_dir = Path(args.output)
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
    # set flag to save ADE20K outputs if requested
    annotator.save_ade20k = args.save_ade20k
    
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
