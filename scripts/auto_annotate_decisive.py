"""
Automatic annotation using OneFormer (ADE20K) - NO PRESERVED VERSION.

Key improvement: All classes are forcibly mapped to Road/Obstacle/Background.
No "preserved" labels - fully automatic.

Usage:
    python scripts/auto_annotate_decisive.py \
        --input data/raw_images \
        --output data/annotations/oneformer_decisive \
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


# =========================================================================
# DECISIVE MAPPING - No Preserved, All Must Be Classified
# =========================================================================
# Strategy: Map everything aggressively
# Unknown items default to Obstacle (safer for autonomous navigation)
# =========================================================================

ADE20K_TO_JETRACER = {
    # FLOOR/ROAD (passable surfaces) → 1
    'floor': 1,
    'flooring': 1,
    'floor, flooring': 1,
    'ground': 1,
    'road': 1,
    'road, route': 1,
    'path': 1,
    'sidewalk': 1,
    'pavement': 1,
    'sidewalk, pavement': 1,
    
    # FLOOR COVERINGS - CRITICAL: These are OBSTACLES
    'rug': 2,          # ← Key fix!
    'carpet': 2,       # ← Key fix!
    'carpeting': 2,
    'carpet, carpeting': 2,
    'mat': 2,
    'cushion': 2,
    'pillow': 2,
    'blanket': 2,      # ← Key fix for user's image!
    'towel': 2,
    'cloth': 2,
    
    # WALLS & STRUCTURE → 2
    'wall': 2,
    'brick wall': 2,
    'wall, brick wall': 2,
    'door': 2,
    'double door': 2,
    'door, double door': 2,
    'fence': 2,
    'fencing': 2,
    'fence, fencing': 2,
    'railing': 2,
    'bannister': 2,
    'stairs': 2,
    'steps': 2,
    'stairway': 2,
    'staircase': 2,
    'step': 2,
    'stair': 2,
    
    # FURNITURE → 2
    'table': 2,
    'chair': 2,
    'sofa': 2,
    'couch': 2,
    'sofa, couch': 2,
    'bed': 2,
    'cabinet': 2,
    'shelf': 2,
    'desk': 2,
    'wardrobe': 2,
    'closet': 2,
    'wardrobe, closet': 2,
    'chest of drawers': 2,
    'chest': 2,
    'drawer': 2,
    
    # OBJECTS → 2
    'box': 2,
    'bag': 2,
    'basket': 2,
    'bottle': 2,
    'book': 2,
    'plant': 2,
    'flower': 2,
    'flowerpot': 2,
    'vase': 2,
    'pot': 2,
    'base': 2,
    
    # BUILDING ELEMENTS → 2
    'building': 2,
    'edifice': 2,
    'building, edifice': 2,
    'house': 2,
    'column': 2,
    'pillar': 2,
    'column, pillar': 2,
    
    # PEOPLE & ANIMALS → 2
    'person': 2,
    'individual': 2,
    'someone': 2,
    'person, individual, someone': 2,
    'animal': 2,
    'dog': 2,
    'cat': 2,
    
    # VEHICLES → 2
    'car': 2,
    'auto': 2,
    'automobile': 2,
    'car, auto, automobile': 2,
    
    # CEILING & SKY → 0 (Background)
    'ceiling': 0,
    'sky': 0,
    
    # WINDOWS & OPENINGS → 0
    'window': 0,
    'windowpane': 0,
    'window, windowpane': 0,
    'glass': 0,
    'mirror': 0,
    
    # LIGHTING → 0
    'light': 0,
    'light source': 0,
    'light, light source': 0,
    'lamp': 0,
    'chandelier': 0,
    
    # SIGNS → 2
    'signboard': 2,
    'sign': 2,
    'signboard, sign': 2,
    'pole': 2,
    'post': 2,
    
    # APPLIANCES → 2
    'fan': 2,
    'air conditioner': 2,
    'heater': 2,
    'radiator': 2,
}


CLASS_COLORS = {
    0: (128, 128, 128),  # Background - Gray
    1: (0, 255, 0),      # Road - Green
    2: (0, 0, 255),      # Obstacle - Red
}

CLASS_NAMES = {0: 'Background', 1: 'Road', 2: 'Obstacle'}


class DecisiveAnnotator:
    """OneFormer annotator with decisive (no-preserved) mapping."""
    
    def __init__(self, model_name: str = "shi-labs/oneformer_ade20k_swin_tiny"):
        print(f"Loading OneFormer: {model_name}")
        print("DECISIVE MODE: No preserved labels, all classifications forced")
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self.processor = OneFormerProcessor.from_pretrained(model_name)
        self.model = OneFormerForUniversalSegmentation.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        print("✓ Model loaded")
        print("  Key mappings:")
        print("    rug/carpet/blanket → Obstacle")
        print("    floor → Road")
        print("    wall → Obstacle")
    
    def _map_ade20k_to_jetracer(
        self,
        ade20k_mask: np.ndarray,
        id2label: Dict[int, str]
    ) -> np.ndarray:
        """
        Map ADE20K segmentation to JetRacer classes (0/1/2 only).
        Unknown classes default to Obstacle (safer).
        """
        h, w = ade20k_mask.shape
        jetracer_mask = np.zeros((h, w), dtype=np.uint8)
        
        # Count unmapped classes for debugging
        unmapped_pixels = 0
        unmapped_classes = set()
        
        for ade20k_id in np.unique(ade20k_mask):
            if ade20k_id not in id2label:
                continue
            
            class_name = id2label[ade20k_id].lower()
            
            # Try exact match first
            if class_name in ADE20K_TO_JETRACER:
                jetracer_class = ADE20K_TO_JETRACER[class_name]
            else:
                # Try fuzzy match (substring)
                matched = False
                for key, value in ADE20K_TO_JETRACER.items():
                    if key in class_name or class_name in key:
                        jetracer_class = value
                        matched = True
                        break
                
                if not matched:
                    # DEFAULT: Unknown → Obstacle (safer for navigation)
                    jetracer_class = 2
                    unmapped_classes.add(class_name)
                    unmapped_pixels += np.sum(ade20k_mask == ade20k_id)
            
            jetracer_mask[ade20k_mask == ade20k_id] = jetracer_class
        
        if unmapped_classes:
            total_pixels = h * w
            unmapped_pct = unmapped_pixels / total_pixels * 100
            print(f"    Unmapped classes (→ Obstacle): {unmapped_pct:.1f}%")
            if unmapped_pct > 10:  # Only show if significant
                print(f"      Classes: {', '.join(sorted(unmapped_classes)[:5])}")
        
        return jetracer_mask
    
    def annotate_image(self, image_path: Path) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Annotate single image."""
        print(f"Processing: {image_path.name}")
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Process
        inputs = self.processor(images=image, task_inputs=["semantic"], return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Inference
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Get segmentation
        predicted_map = self.processor.post_process_semantic_segmentation(
            outputs,
            target_sizes=[image.size[::-1]]
        )[0]
        
        ade20k_mask = predicted_map.cpu().numpy()
        
        # Map to JetRacer classes
        id2label = self.model.config.id2label
        jetracer_mask = self._map_ade20k_to_jetracer(ade20k_mask, id2label)
        
        # Statistics
        stats = {}
        for class_id, class_name in CLASS_NAMES.items():
            count = np.sum(jetracer_mask == class_id)
            pct = count / jetracer_mask.size * 100
            stats[class_name.lower()] = {'count': int(count), 'percentage': float(pct)}
        
        print(f"  Distribution:")
        for class_name, data in stats.items():
            marker = ''
            if class_name == 'obstacle' and data['percentage'] > 25:
                marker = ' ✅'
            elif class_name == 'road' and 40 <= data['percentage'] <= 70:
                marker = ' ✅'
            print(f"    {class_name.capitalize():12s}: {data['percentage']:5.1f}%{marker}")
        
            return np.array(image), jetracer_mask, ade20k_mask, stats


def main():
    parser = argparse.ArgumentParser(
        description='Decisive OneFormer annotation (no preserved labels)'
    )
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--vehicle-mask', type=str, default=None,
                       help='Optional vehicle mask to exclude')
    
    args = parser.parse_args()
    
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    
    # Create output dirs
    masks_dir = output_dir / 'masks'
    masks_dir.mkdir(parents=True, exist_ok=True)
    
    if args.visualize:
        vis_dir = output_dir / 'visualizations'
        vis_dir.mkdir(parents=True, exist_ok=True)
    
    # Load vehicle mask if provided
    vehicle_mask = None
    if args.vehicle_mask:
        vehicle_mask_path = Path(args.vehicle_mask)
        if vehicle_mask_path.exists():
            vehicle_mask = cv2.imread(str(vehicle_mask_path), cv2.IMREAD_GRAYSCALE)
            vehicle_mask = (vehicle_mask > 127).astype(np.uint8)
            print(f"✓ Loaded vehicle mask: {vehicle_mask_path}")
    
    print("=" * 60)
    print("Decisive OneFormer Annotation (No Preserved)")
    print("=" * 60)
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    print("=" * 60)
    print()
    
    # Initialize annotator
    annotator = DecisiveAnnotator()
    
    # Find images
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        image_files.extend(list(input_dir.rglob(ext)))
    
    print(f"Found {len(image_files)} images\n")
    
    # Process images
    for image_path in sorted(image_files):
        try:
            image, mask, ade20k_mask, stats = annotator.annotate_image(image_path)

            # Apply vehicle mask if available
            if vehicle_mask is not None:
                mask_resized = cv2.resize(vehicle_mask, (mask.shape[1], mask.shape[0]))
                # mask_resized: 1 = vehicle, 0 = non-vehicle
                # Ensure we set vehicle region to Background (0)
                mask[mask_resized == 1] = 0

            # Save mask
            mask_path = masks_dir / (image_path.stem + '_mask.png')
            cv2.imwrite(str(mask_path), mask)

            # Visualization
            if args.visualize:
                # Build ADE20K color map for obstacle visualization
                ade_color = {}
                for uid in np.unique(ade20k_mask):
                    hval = (int(uid) * 37) % 180
                    hsv = np.uint8([[[hval, 200, 200]]])
                    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0].tolist()
                    ade_color[int(uid)] = tuple(int(x) for x in bgr)

                colored_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
                # Background: gray
                colored_mask[mask == 0] = CLASS_COLORS[0]
                # Road: keep green
                colored_mask[mask == 1] = CLASS_COLORS[1]
                # Obstacle: color by ADE20K label
                obstacle_idx = (mask == 2)
                for uid, col in ade_color.items():
                    colored_mask[np.logical_and(obstacle_idx, ade20k_mask == uid)] = col

                # Blend with original (image is RGB; convert to BGR for OpenCV blending)
                image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                colored_mask_bgr = colored_mask[:, :, ::-1]
                vis_bgr = cv2.addWeighted(image_bgr, 0.5, colored_mask_bgr, 0.5, 0)

                # Draw vehicle mask boundary (vehicle region in mask_resized==1)
                if vehicle_mask is not None:
                    contours, _ = cv2.findContours(
                        mask_resized.astype(np.uint8),
                        cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE
                    )
                    cv2.drawContours(vis_bgr, contours, -1, (255, 255, 0), 2)

                vis_path = vis_dir / (image_path.stem + '_vis.jpg')
                cv2.imwrite(str(vis_path), vis_bgr)

            print(f"  ✓ Saved\n")

        except Exception as e:
            print(f"  ✗ Error: {e}\n")
            import traceback
            traceback.print_exc()

    print("=" * 60)
    print("Decisive Annotation Complete!")
    print("=" * 60)
    print(f"Masks: {masks_dir}")
    if args.visualize:
        print(f"Visualizations: {vis_dir}")
    print("=" * 60)

    return 0


if __name__ == '__main__':
    exit(main())
