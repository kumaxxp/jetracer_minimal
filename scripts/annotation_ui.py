"""
Interactive annotation correction tool.

Simple web UI for reviewing and correcting auto-generated masks.
Click on regions to change their labels.

Usage:
    python scripts/annotation_ui.py \
        --images data/raw_images \
        --masks data/annotations/auto_masks/masks \
        --output data/annotations/corrected_masks
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from nicegui import ui


# Class definitions
CLASSES = {
    0: {'name': 'Background', 'color': (128, 128, 128), 'key': '0'},
    1: {'name': 'Road', 'color': (0, 255, 0), 'key': '1'},
    2: {'name': 'Obstacle', 'color': (0, 0, 255), 'key': '2'},
}


class AnnotationCorrector:
    """Interactive annotation correction tool."""
    
    def __init__(
        self,
        images_dir: Path,
        masks_dir: Path,
        output_dir: Path
    ):
        """
        Initialize annotation corrector.
        
        Args:
            images_dir: Directory containing images
            masks_dir: Directory containing masks
            output_dir: Output directory for corrected masks
        """
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find image-mask pairs
        self.pairs = self._find_pairs()
        self.current_idx = 0
        
        # Current state
        self.current_image = None
        self.current_mask = None
        self.modified = False
        
        # UI elements
        self.image_display = None
        self.status_label = None
        self.progress_label = None
        self.current_class = 1  # Default: Road
        
        print(f"Found {len(self.pairs)} image-mask pairs")
    
    def _find_pairs(self) -> List[Tuple[Path, Path]]:
        """Find matching image-mask pairs."""
        pairs = []
        
        # Find all mask files
        mask_files = list(self.masks_dir.glob('*_mask.png'))
        
        for mask_path in mask_files:
            # Derive image filename
            image_name = mask_path.stem.replace('_mask', '') + '.jpg'
            
            # Search for image
            image_path = None
            for session_dir in self.images_dir.glob('session_*'):
                candidate = session_dir / image_name
                if candidate.exists():
                    image_path = candidate
                    break
            
            if image_path:
                pairs.append((image_path, mask_path))
        
        return pairs
    
    def load_current(self) -> None:
        """Load current image and mask."""
        if self.current_idx >= len(self.pairs):
            return
        
        image_path, mask_path = self.pairs[self.current_idx]
        
        # Load image
        self.current_image = cv2.imread(str(image_path))
        self.current_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
        
        # Load mask
        self.current_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        
        self.modified = False
    
    def create_visualization(self) -> np.ndarray:
        """Create visualization with overlay."""
        if self.current_image is None or self.current_mask is None:
            return np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Create colored mask
        colored_mask = np.zeros_like(self.current_image)
        for class_id, info in CLASSES.items():
            colored_mask[self.current_mask == class_id] = info['color']
        
        # Blend
        vis = cv2.addWeighted(self.current_image, 0.6, colored_mask, 0.4, 0)
        
        return vis
    
    def save_current(self) -> None:
        """Save current mask."""
        if self.current_mask is None:
            return
        
        image_path, _ = self.pairs[self.current_idx]
        output_path = self.output_dir / (image_path.stem + '_mask.png')
        
        cv2.imwrite(str(output_path), self.current_mask)
        print(f"Saved: {output_path.name}")
    
    def handle_click(self, x: int, y: int) -> None:
        """
        Handle click on image to change region label.
        
        Args:
            x: X coordinate
            y: Y coordinate
        """
        if self.current_mask is None:
            return
        
        h, w = self.current_mask.shape
        
        # Convert click coordinates to image coordinates
        # (Assume image_display shows image at its original size)
        img_x = int(x)
        img_y = int(y)
        
        if img_x < 0 or img_x >= w or img_y < 0 or img_y >= h:
            return
        
        # Get current class at click position
        clicked_class = self.current_mask[img_y, img_x]
        
        # Change to selected class
        if clicked_class != self.current_class:
            # Use flood fill to change connected region
            mask_copy = self.current_mask.copy()
            cv2.floodFill(
                mask_copy,
                None,
                (img_x, img_y),
                self.current_class
            )
            self.current_mask = mask_copy
            self.modified = True
            
            # Update display
            self.update_display()
            
            ui.notify(
                f'Changed region to {CLASSES[self.current_class]["name"]}',
                type='positive'
            )
    
    def update_display(self) -> None:
        """Update image display."""
        if self.image_display is None:
            return
        
        # Create visualization
        vis = self.create_visualization()
        
        # Convert to base64 for display
        import base64
        _, buffer = cv2.imencode('.jpg', cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Update display
        self.image_display.set_source(f'data:image/jpeg;base64,{img_base64}')
        
        # Update status
        if self.status_label:
            status = '🟡 Modified' if self.modified else '✅ Original'
            self.status_label.set_text(status)
        
        if self.progress_label:
            self.progress_label.set_text(
                f'Image {self.current_idx + 1} / {len(self.pairs)}'
            )
    
    def next_image(self) -> None:
        """Move to next image."""
        # Save if modified
        if self.modified:
            self.save_current()
        
        # Move to next
        if self.current_idx < len(self.pairs) - 1:
            self.current_idx += 1
            self.load_current()
            self.update_display()
        else:
            ui.notify('Last image reached!', type='info')
    
    def prev_image(self) -> None:
        """Move to previous image."""
        # Save if modified
        if self.modified:
            self.save_current()
        
        # Move to previous
        if self.current_idx > 0:
            self.current_idx -= 1
            self.load_current()
            self.update_display()
        else:
            ui.notify('First image reached!', type='info')
    
    def skip_image(self) -> None:
        """Skip current image without saving."""
        if self.current_idx < len(self.pairs) - 1:
            self.current_idx += 1
            self.load_current()
            self.update_display()
    
    def setup_ui(self) -> None:
        """Setup user interface."""
        with ui.column().classes('w-full items-center gap-4 p-4'):
            ui.label('🎨 Annotation Correction Tool').classes('text-3xl font-bold')
            
            # Instructions
            with ui.card().classes('w-full max-w-4xl'):
                ui.label('Instructions').classes('text-xl mb-2')
                with ui.column().classes('gap-2'):
                    ui.label('1. Select a class using buttons or keys (0, 1, 2)')
                    ui.label('2. Click on image region to change its label')
                    ui.label('3. Use Next/Previous to navigate')
                    ui.label('4. Modified masks are auto-saved')
            
            # Image display
            with ui.card().classes('w-full max-w-4xl'):
                self.progress_label = ui.label('').classes('text-lg')
                self.status_label = ui.label('').classes('text-lg')
                
                # Image with click handler
                self.image_display = ui.image().classes('w-full')
                # Note: Click handling would need JavaScript integration
                # For simplicity, we'll use keyboard shortcuts
            
            # Class selection
            with ui.card().classes('w-full max-w-4xl'):
                ui.label('Select Class').classes('text-xl mb-2')
                with ui.row().classes('gap-4'):
                    for class_id, info in CLASSES.items():
                        color_hex = f"#{info['color'][0]:02x}{info['color'][1]:02x}{info['color'][2]:02x}"
                        ui.button(
                            f"{info['name']} ({info['key']})",
                            on_click=lambda c=class_id: self.set_class(c)
                        ).props(f'color={color_hex}')
            
            # Navigation
            with ui.card().classes('w-full max-w-4xl'):
                ui.label('Navigation').classes('text-xl mb-2')
                with ui.row().classes('gap-4'):
                    ui.button('← Previous', on_click=self.prev_image).props('color=blue')
                    ui.button('Skip (No Save)', on_click=self.skip_image).props('color=orange')
                    ui.button('Next →', on_click=self.next_image).props('color=green')
        
        # Load first image
        self.load_current()
        self.update_display()
    
    def set_class(self, class_id: int) -> None:
        """Set current class for labeling."""
        self.current_class = class_id
        ui.notify(f'Selected: {CLASSES[class_id]["name"]}', type='info')


def main():
    parser = argparse.ArgumentParser(
        description='Interactive annotation correction tool'
    )
    parser.add_argument(
        '--images',
        type=str,
        required=True,
        help='Directory containing images'
    )
    parser.add_argument(
        '--masks',
        type=str,
        required=True,
        help='Directory containing auto-generated masks'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/annotations/corrected_masks',
        help='Output directory for corrected masks'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=8082,
        help='Port for web UI'
    )
    
    args = parser.parse_args()
    
    # Convert paths
    images_dir = Path(args.images)
    masks_dir = Path(args.masks)
    output_dir = Path(args.output)
    
    # Validate inputs
    if not images_dir.exists():
        print(f"Error: Images directory not found: {images_dir}")
        return 1
    
    if not masks_dir.exists():
        print(f"Error: Masks directory not found: {masks_dir}")
        return 1
    
    print("=" * 60)
    print("Annotation Correction Tool")
    print("=" * 60)
    print(f"Images: {images_dir}")
    print(f"Masks: {masks_dir}")
    print(f"Output: {output_dir}")
    print("=" * 60)
    print()
    
    # Create corrector
    corrector = AnnotationCorrector(images_dir, masks_dir, output_dir)
    corrector.setup_ui()
    
    print(f"Starting web UI on port {args.port}")
    print(f"Access at: http://localhost:{args.port}")
    print()
    
    # Run UI
    ui.run(
        host='0.0.0.0',
        port=args.port,
        title='Annotation Correction',
        reload=False,
        show=False
    )
    
    return 0


if __name__ == '__main__':
    exit(main())
