"""
Interactive Object Annotation Tool for JetRacer

This tool allows users to:
1. View OneFormer detections
2. Click on detected objects
3. Toggle them as "valid obstacles" or "ignorable"
4. Visualize with striped pattern for ignorable objects
5. Save metadata for training

Usage:
    python scripts/annotation_tool_interactive.py \
        --session data/annotations/oneformer_decisive/session_20251127_153358 \
        --port 8083
"""

from __future__ import annotations

import argparse
import colorsys
import json
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from flask import Flask, render_template_string, jsonify, request, send_file
from flask_cors import CORS


# HTML Template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>JetRacer Object Annotation Tool</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f0f0f0;
        }
        .container {
            display: flex;
            gap: 20px;
            max-width: 1400px;
            margin: 0 auto;
        }
        .left-panel {
            flex: 2;
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .right-panel {
            flex: 1;
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        h1 {
            margin: 0 0 20px 0;
            font-size: 24px;
            color: #333;
        }
        .session-info {
            margin-bottom: 20px;
            padding: 10px;
            background: #f8f8f8;
            border-radius: 4px;
            font-size: 14px;
        }
        #canvas {
            border: 2px solid #ddd;
            cursor: crosshair;
            max-width: 100%;
            display: block;
        }
        .controls {
            margin-top: 20px;
            display: flex;
            gap: 10px;
            justify-content: center;
        }
        button {
            padding: 10px 20px;
            font-size: 14px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            background: #4CAF50;
            color: white;
            transition: background 0.3s;
        }
        button:hover {
            background: #45a049;
        }
        button:disabled {
            background: #ccc;
            cursor: not-allowed;
        }
        .object-list {
            max-height: 500px;
            overflow-y: auto;
        }
        .object-item {
            padding: 10px;
            margin-bottom: 8px;
            border: 1px solid #ddd;
            border-radius: 4px;
            cursor: pointer;
            transition: all 0.2s;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        .object-item:hover {
            background: #f8f8f8;
        }
        .object-item.selected {
            border-color: #4CAF50;
            background: #e8f5e9;
        }
        .object-item.disabled {
            opacity: 0.6;
        }
        .object-info {
            flex: 1;
        }
        .object-name {
            font-weight: bold;
            color: #333;
        }
        .object-stats {
            font-size: 12px;
            color: #666;
            margin-top: 4px;
        }
        .toggle-switch {
            width: 50px;
            height: 24px;
            background: #ccc;
            border-radius: 12px;
            position: relative;
            cursor: pointer;
            transition: background 0.3s;
        }
        .toggle-switch.active {
            background: #4CAF50;
        }
        .toggle-switch .slider {
            width: 20px;
            height: 20px;
            background: white;
            border-radius: 50%;
            position: absolute;
            top: 2px;
            left: 2px;
            transition: left 0.3s;
        }
        .toggle-switch.active .slider {
            left: 28px;
        }
        .legend {
            margin-top: 20px;
            padding: 15px;
            background: #f8f8f8;
            border-radius: 4px;
        }
        .legend-item {
            display: flex;
            align-items: center;
            margin-bottom: 8px;
        }
        .legend-color {
            width: 30px;
            height: 20px;
            margin-right: 10px;
            border: 1px solid #ddd;
        }
        .legend-road { background: rgba(0, 255, 0, 0.5); }
        .legend-obstacle { background: rgba(255, 0, 0, 0.5); }
        .legend-ignorable {
            background: repeating-linear-gradient(
                45deg,
                rgba(255, 255, 0, 0.3),
                rgba(255, 255, 0, 0.3) 10px,
                rgba(255, 255, 255, 0.3) 10px,
                rgba(255, 255, 255, 0.3) 20px
            );
        }
        .shortcuts {
            margin-top: 20px;
            font-size: 12px;
            color: #666;
            padding: 10px;
            background: #f8f8f8;
            border-radius: 4px;
        }
        .shortcuts h3 {
            margin: 0 0 10px 0;
            font-size: 14px;
        }
        .shortcuts div {
            margin-bottom: 5px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="left-panel">
            <h1>JetRacer Object Annotation Tool</h1>
            <div class="session-info">
                <div><strong>Session:</strong> <span id="session-name"></span></div>
                <div><strong>Image:</strong> <span id="image-name"></span> (<span id="image-index"></span>)</div>
            </div>
            <canvas id="canvas"></canvas>
            <div class="controls">
                <button id="prev-btn" onclick="navigate(-1)">◀ Previous</button>
                <button id="save-btn" onclick="save()">💾 Save</button>
                <button id="reset-btn" onclick="reset()">🔄 Reset</button>
                <button id="next-btn" onclick="navigate(1)">Next ▶</button>
            </div>
            <div class="legend">
                <h3>Visualization</h3>
                <div class="legend-item">
                    <div class="legend-color" style="background: linear-gradient(to right, #ff6b6b, #4ecdc4, #45b7d1, #f7b731);"></div>
                    <span>🎨 ADE-colored segments</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color legend-ignorable"></div>
                    <span>▦ ROAD (striped pattern)</span>
                </div>
                <div style="margin-top: 10px; font-size: 12px; color: #666;">
                    <strong>How to use:</strong><br>
                    1. Click on segment to toggle ROAD<br>
                    2. ROAD segments get stripes<br>
                    3. Changes apply to ALL images<br>
                    4. Save to create mapping table
                </div>
            </div>
        </div>
        <div class="right-panel">
            <h2>ROAD Labels</h2>
            <div style="font-size: 13px; color: #666; margin-bottom: 15px; padding: 10px; background: #f8f8f8; border-radius: 4px;">
                <strong>Click on segment</strong> to toggle ROAD status<br>
                <em>ROAD labels show striped pattern</em><br>
                <em>Changes apply to ALL images</em>
            </div>
            <div class="road-labels-list" id="road-labels-list"></div>
            
            <h3 style="margin-top: 30px;">All Labels in Current Image</h3>
            <div class="object-list" id="object-list"></div>
            <div class="shortcuts">
                <h3>⌨️ Keyboard Shortcuts</h3>
                <div><strong>Space:</strong> Toggle impassable ⇄ ignorable</div>
                <div><strong>→:</strong> Next image</div>
                <div><strong>←:</strong> Previous image</div>
                <div><strong>S:</strong> Save</div>
                <div style="margin-top: 8px; font-size: 11px; color: #888;">
                    💡 Tip: Checked = Impassable (must avoid)<br>
                    💡 Unchecked = Ignorable (can touch)
                </div>
            </div>
        </div>
    </div>

    <script>
        let currentIndex = 0;
        let images = [];
        let currentLabels = [];
        let roadLabels = [];
        const canvas = document.getElementById('canvas');
        const ctx = canvas.getContext('2d');

        // Load session data
        async function loadSession() {
            const response = await fetch('/api/session');
            const data = await response.json();
            images = data.images;
            document.getElementById('session-name').textContent = data.session_name;
            loadImage(0);
        }

        // Load specific image
        async function loadImage(index) {
            if (index < 0 || index >= images.length) return;
            
            currentIndex = index;
            
            // Update UI
            document.getElementById('image-name').textContent = images[index];
            document.getElementById('image-index').textContent = `${index + 1}/${images.length}`;
            document.getElementById('prev-btn').disabled = index === 0;
            document.getElementById('next-btn').disabled = index === images.length - 1;
            
            // Load image data
            const response = await fetch(`/api/image/${index}`);
            const data = await response.json();
            
            // Load image
            const img = new Image();
            img.onload = () => {
                canvas.width = img.width;
                canvas.height = img.height;
                drawImage(data);
            };
            img.src = data.image_url;
            
            // Store labels
            currentLabels = data.labels;
            roadLabels = data.road_labels;
            renderLabelLists();
        }

        // Draw image with overlays
        function drawImage(data) {
            const img = new Image();
            img.onload = () => {
                // Draw base image
                ctx.drawImage(img, 0, 0);
                
                // Draw overlay
                const overlayImg = new Image();
                overlayImg.onload = () => {
                    ctx.globalAlpha = 0.6;
                    ctx.drawImage(overlayImg, 0, 0);
                    ctx.globalAlpha = 1.0;
                };
                overlayImg.src = data.overlay_url;
            };
            img.src = data.image_url;
        }

        // Render label lists
        function renderLabelLists() {
            // Render ROAD labels list
            const roadList = document.getElementById('road-labels-list');
            roadList.innerHTML = '';
            
            if (roadLabels.length === 0) {
                roadList.innerHTML = '<div style="color: #999; font-style: italic;">No ROAD labels yet. Click on segments to add.</div>';
            } else {
                roadLabels.forEach(label => {
                    const item = document.createElement('div');
                    item.className = 'road-label-item';
                    item.style.cssText = 'padding: 8px; margin-bottom: 4px; background: #e8f5e9; border-left: 3px solid #4CAF50; cursor: pointer;';
                    item.onclick = () => toggleADELabel(label.ade_id);
                    item.innerHTML = `
                        <div style="font-weight: bold;">▦ ${label.label}</div>
                        <div style="font-size: 11px; color: #666;">ADE-${label.ade_id} • Click to remove</div>
                    `;
                    roadList.appendChild(item);
                });
            }
            
            // Render all labels in current image
            const objectList = document.getElementById('object-list');
            objectList.innerHTML = '';
            
            currentLabels.forEach(label => {
                const item = document.createElement('div');
                item.className = 'object-item';
                item.style.cssText = 'padding: 10px; margin-bottom: 8px; border: 1px solid #ddd; border-radius: 4px; cursor: pointer;';
                if (label.is_road) {
                    item.style.background = '#e8f5e9';
                    item.style.borderColor = '#4CAF50';
                }
                item.onclick = () => toggleADELabel(label.ade_id);
                
                const badge = label.is_road ? '▦' : '⬛';
                const status = label.is_road ? 'ROAD' : 'Obstacle';
                
                item.innerHTML = `
                    <div style="font-weight: bold;">${badge} ${label.label}</div>
                    <div style="font-size: 12px; color: #666;">${label.percentage.toFixed(1)}% • ADE-${label.ade_id} • ${status}</div>
                `;
                objectList.appendChild(item);
            });
        }

        // Toggle ADE label
        async function toggleADELabel(adeId) {
            const response = await fetch(`/api/toggle/${adeId}`, {
                method: 'POST'
            });
            const data = await response.json();
            roadLabels = data.road_labels;
            
            // Reload current image to update visualization
            loadImage(currentIndex);
        }

        // Navigation
        function navigate(delta) {
            loadImage(currentIndex + delta);
        }

        // Save
        async function save() {
            const response = await fetch('/api/save', {
                method: 'POST'
            });
            const data = await response.json();
            alert(data.message);
        }

        // Reset
        async function reset() {
            if (confirm('Reset ALL ROAD mappings? This affects the entire dataset.')) {
                const response = await fetch(`/api/reset`, {
                    method: 'POST'
                });
                loadImage(currentIndex);
            }
        }

        // Canvas click handler
        canvas.addEventListener('click', async (e) => {
            const rect = canvas.getBoundingClientRect();
            const x = Math.floor((e.clientX - rect.left) * canvas.width / rect.width);
            const y = Math.floor((e.clientY - rect.top) * canvas.height / rect.height);
            
            const response = await fetch(`/api/pick/${currentIndex}/${x}/${y}`);
            const data = await response.json();
            
            if (data.ade_id !== null && data.ade_id !== undefined) {
                toggleADELabel(data.ade_id);
            }
        });

        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.key === 'ArrowRight') {
                navigate(1);
            } else if (e.key === 'ArrowLeft') {
                navigate(-1);
            } else if (e.key === 's' || e.key === 'S') {
                e.preventDefault();
                save();
            } else if (e.key === 'r' || e.key === 'R') {
                e.preventDefault();
                reset();
            }
        });

        // Initialize
        loadSession();
    </script>
</body>
</html>
"""


class InteractiveAnnotationTool:
    """
    Interactive ADE20K to JetRacer Mapping Tool.
    
    Purpose: Create consistent ADE20K → ROAD mapping across entire dataset.
    
    Workflow:
    1. Display image with ADE-colored segmentation
    2. User clicks on segment to toggle ROAD classification
    3. Clicked ADE label becomes ROAD for ALL images
    4. Striped pattern shows ROAD segments
    5. Right panel shows list of ROAD labels
    6. Save ADE20K_TO_JETRACER mapping table
    """
    
    # ADE20K common labels (subset of 150 classes)
    ADE20K_LABELS = {
        1: 'wall',
        3: 'floor',
        4: 'ceiling',
        5: 'door',
        7: 'table',
        8: 'windowpane',
        10: 'chair',
        11: 'car',
        12: 'person',
        13: 'curtain',
        14: 'painting',
        15: 'sofa',
        16: 'bed',
        18: 'cabinet',
        19: 'desk',
        22: 'armchair',
        23: 'seat',
        24: 'fence',
        25: 'pillow',
        28: 'rug',
        29: 'lamp',
        30: 'bathtub',
        31: 'railing',
        32: 'cushion',
        33: 'box',
        34: 'column',
        35: 'signboard',
        36: 'chest of drawers',
        37: 'counter',
        38: 'sink',
        39: 'fireplace',
        40: 'refrigerator',
        41: 'stairs',
        42: 'escalator',
        43: 'bookcase',
        44: 'book',
        45: 'blind',
        46: 'shelf',
        47: 'stairway',
        48: 'ottoman',
        49: 'bottle',
        50: 'buffet',
        51: 'poster',
        52: 'stage',
        53: 'van',
        54: 'ship',
        55: 'fountain',
        56: 'awning',
        57: 'streetlight',
        58: 'truck',
        59: 'tower',
        60: 'chandelier',
        61: 'canopy',
        62: 'washer',
        63: 'plaything',
        64: 'pool table',
        65: 'stool',
        66: 'barrel',
        67: 'basket',
        68: 'bag',
        69: 'minibike',
        70: 'cradle',
        71: 'oven',
        72: 'ball',
        73: 'food',
        74: 'step',
        75: 'tank',
        76: 'trade name',
        77: 'microwave',
        78: 'pot',
        79: 'animal',
        80: 'bicycle',
        81: 'dishwasher',
        82: 'screen',
        83: 'blanket',
        84: 'sculpture',
        85: 'hood',
        86: 'sconce',
        87: 'vase',
        88: 'traffic light',
        89: 'tray',
        90: 'ashcan',
        91: 'fan',
        92: 'pier',
        93: 'screen door',
        94: 'plate',
        95: 'monitor',
        96: 'bulletin board',
        97: 'shower',
        98: 'radiator',
        99: 'glass',
        100: 'clock',
    }
    
    def __init__(self, session_dir: Path):
        # Resolve session path to absolute so file I/O is stable regardless of CWD
        self.session_dir = Path(session_dir).resolve()
        self.masks_dir = self.session_dir / 'masks'
        self.images_dir = self.session_dir.parent.parent.parent / 'raw_images' / self.session_dir.name
        self.ade_masks_dir = self.session_dir / 'ade20k_masks'
        self.output_dir = self.session_dir / 'labeled'
        # Ensure the labeled directory exists (create parents if necessary)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load images
        self.image_files = sorted(list(self.images_dir.glob('*.jpg')))
        
        # Load or initialize ADE20K → ROAD mapping
        self.mapping_path = self.output_dir / 'ade_to_road_mapping.json'
        self.ade_to_road = self._load_mapping()
        
        # Detect all ADE labels in dataset
        self.all_ade_labels = self._detect_all_ade_labels()
        
        print(f"Loaded session: {self.session_dir.name}")
        print(f"Found {len(self.image_files)} images")
        print(f"Detected {len(self.all_ade_labels)} unique ADE labels")
    
    def _load_mapping(self) -> Dict[int, bool]:
        """
        Load ADE20K → ROAD mapping.
        
        Returns:
            Dict[ade_id, is_road]
            - True: This ADE label is ROAD
            - False: This ADE label is Obstacle
        """
        if self.mapping_path.exists():
            with open(self.mapping_path) as f:
                data = json.load(f)
                # Convert string keys back to int
                return {int(k): v for k, v in data.items()}
        else:
            return {}
    
    def _detect_all_ade_labels(self) -> List[int]:
        """Detect all unique ADE labels across entire dataset."""
        all_labels = set()
        
        for img_file in self.image_files:
            ade_mask_path = self.ade_masks_dir / (img_file.stem + '_ade20k.png')
            if ade_mask_path.exists():
                ade_mask = cv2.imread(str(ade_mask_path), cv2.IMREAD_GRAYSCALE)
                all_labels.update(np.unique(ade_mask).tolist())
        
        return sorted(list(all_labels))
    
    def save_mapping(self):
        """Save ADE20K → ROAD mapping to disk."""
        with open(self.mapping_path, 'w') as f:
            json.dump(self.ade_to_road, f, indent=2)
        print(f"Saved mapping: {self.mapping_path}")
        print(f"  ROAD labels: {sum(self.ade_to_road.values())}")
        print(f"  Obstacle labels: {len(self.ade_to_road) - sum(self.ade_to_road.values())}")
    
    def get_ade_labels_in_image(self, image_index: int) -> List[Dict]:
        """
        Get ADE labels present in current image.
        
        Returns list with:
        - ade_id: ADE20K class ID
        - label: ADE20K class name
        - pixels: number of pixels
        - percentage: percentage of image
        - is_road: whether this label is classified as ROAD
        """
        img_file = self.image_files[image_index]
        
        # Load ADE20K mask
        ade_mask_path = self.ade_masks_dir / (img_file.stem + '_ade20k.png')
        ade_mask = cv2.imread(str(ade_mask_path), cv2.IMREAD_GRAYSCALE)
        
        # Get unique ADE IDs
        labels = []
        for ade_id in np.unique(ade_mask):
            pixel_count = np.sum(ade_mask == ade_id)
            
            if pixel_count > 100:  # Minimum size threshold
                # Check if this label is ROAD
                is_road = self.ade_to_road.get(int(ade_id), False)
                
                # Get label name
                label_name = self.ADE20K_LABELS.get(int(ade_id), f"Unknown-{ade_id}")
                
                labels.append({
                    'ade_id': int(ade_id),
                    'label': label_name,
                    'pixels': int(pixel_count),
                    'percentage': float(pixel_count / ade_mask.size * 100),
                    'is_road': is_road
                })
        
        # Sort by pixel count
        labels.sort(key=lambda x: x['pixels'], reverse=True)
        
        return labels
    
    def get_road_labels(self) -> List[Dict]:
        """
        Get all ADE labels that are classified as ROAD.
        
        Returns list for display in right panel.
        """
        road_labels = []
        
        for ade_id, is_road in self.ade_to_road.items():
            if is_road:
                label_name = self.ADE20K_LABELS.get(ade_id, f"Unknown-{ade_id}")
                road_labels.append({
                    'ade_id': ade_id,
                    'label': label_name
                })
        
        # Sort by name
        road_labels.sort(key=lambda x: x['label'])
        
        return road_labels
    
    def toggle_road_label(self, ade_id: int):
        """
        Toggle ADE label between ROAD and Obstacle.
        
        This affects ALL images in the dataset.
        """
        current = self.ade_to_road.get(ade_id, False)
        self.ade_to_road[ade_id] = not current
        
        label_name = self.ADE20K_LABELS.get(ade_id, f"Unknown-{ade_id}")
        status = "ROAD" if not current else "Obstacle"
        print(f"Toggled {label_name} (ADE-{ade_id}) → {status}")
    
    def _ade_id_to_color(self, ade_id: int) -> Tuple[int, int, int]:
        """
        Convert ADE20K ID to unique color using HSV.
        
        Returns RGB color tuple.
        """
        # Use HSV to generate distinct colors
        import colorsys
        
        # Map ADE ID to hue (0-360 degrees)
        hue = (ade_id * 137.508) % 360  # Golden angle for better distribution
        saturation = 0.7
        value = 0.9
        
        # Convert HSV to RGB
        r, g, b = colorsys.hsv_to_rgb(hue / 360.0, saturation, value)
        
        return (int(b * 255), int(g * 255), int(r * 255))  # BGR for OpenCV
    
    def create_overlay(self, image_index: int) -> np.ndarray:
        """
        Create overlay visualization.
        
        Display:
        - ADE-colored segmentation for all labels
        - Striped pattern overlay for ROAD labels
        
        Logic:
        - All segments: ADE-specific unique color
        - ROAD segments: Add white stripes on top
        - Non-ROAD segments: Just ADE color
        """
        img_file = self.image_files[image_index]
        
        # Load ADE mask
        ade_mask_path = self.ade_masks_dir / (img_file.stem + '_ade20k.png')
        ade_mask = cv2.imread(str(ade_mask_path), cv2.IMREAD_GRAYSCALE)
        
        h, w = ade_mask.shape
        overlay = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Get all ADE labels in this image
        labels = self.get_ade_labels_in_image(image_index)
        
        for label in labels:
            ade_id = label['ade_id']
            # Get pixels for this ADE label
            mask = (ade_mask == ade_id)
            # Get unique color for this ADE ID (BGR)
            color = self._ade_id_to_color(ade_id)
            # Apply color to overlay
            overlay[mask] = color

        # Blend overlay with original image so ADE colors are visible on top of photo
        img_bgr = cv2.imread(str(img_file))
        if img_bgr is None:
            return overlay

        vis = cv2.addWeighted(img_bgr, 0.5, overlay, 0.5, 0)

        # Draw black horizontal stripe pattern for ROAD labels on top of blended image
        stripe_width = 8
        stripe_gap = 12
        stripe_period = stripe_width + stripe_gap
        for label in labels:
            ade_id = label['ade_id']
            is_road = label['is_road']
            if not is_road:
                continue
            mask = (ade_mask == ade_id)
            for y in range(h):
                if ((y // stripe_period) % 2) == 0:
                    cols = np.where(mask[y])[0]
                    if cols.size:
                        vis[y, cols] = (0, 0, 0)

        return vis


# Flask app
app = Flask(__name__)
CORS(app)

tool = None


@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route('/api/session')
def api_session():
    return jsonify({
        'session_name': tool.session_dir.name,
        'images': [f.name for f in tool.image_files]
    })


@app.route('/api/image/<int:index>')
def api_image(index):
    img_file = tool.image_files[index]
    labels = tool.get_ade_labels_in_image(index)
    road_labels = tool.get_road_labels()
    
    return jsonify({
        'image_url': f'/image/{index}',
        'overlay_url': f'/overlay/{index}',
        'labels': labels,
        'road_labels': road_labels
    })


@app.route('/image/<int:index>')
def serve_image(index):
    return send_file(tool.image_files[index])


@app.route('/overlay/<int:index>')
def serve_overlay(index):
    overlay = tool.create_overlay(index)
    
    # Save to temp file
    temp_path = tool.output_dir / f'temp_overlay_{index}.png'
    # Overlay is already in BGR order (see _ade_id_to_color), write directly
    cv2.imwrite(str(temp_path), overlay)
    
    return send_file(temp_path)


@app.route('/api/toggle/<int:ade_id>', methods=['POST'])
def api_toggle(ade_id):
    """Toggle ADE label between ROAD and Obstacle."""
    tool.toggle_road_label(ade_id)
    
    return jsonify({
        'road_labels': tool.get_road_labels()
    })


@app.route('/api/pick/<int:index>/<int:x>/<int:y>')
def api_pick(index, x, y):
    """Pick ADE label at pixel coordinates."""
    img_file = tool.image_files[index]
    
    # Load ADE mask
    ade_mask_path = tool.ade_masks_dir / (img_file.stem + '_ade20k.png')
    ade_mask = cv2.imread(str(ade_mask_path), cv2.IMREAD_GRAYSCALE)
    
    # Get ADE ID at pixel
    ade_id = int(ade_mask[y, x])
    
    return jsonify({'ade_id': ade_id})


@app.route('/api/save', methods=['POST'])
def api_save():
    tool.save_mapping()
    return jsonify({'message': 'ADE20K → ROAD mapping saved successfully!'})


@app.route('/api/reset', methods=['POST'])
def api_reset():
    """Reset all mappings."""
    tool.ade_to_road = {}
    return jsonify({'message': 'All mappings reset', 'road_labels': []})


def main():
    parser = argparse.ArgumentParser(
        description='Interactive Object Annotation Tool'
    )
    parser.add_argument('--session', type=str, required=True,
                       help='Path to OneFormer session directory')
    parser.add_argument('--port', type=int, default=8083,
                       help='Port number')
    
    args = parser.parse_args()
    
    global tool
    tool = InteractiveAnnotationTool(args.session)
    
    print("=" * 60)
    print("Interactive Object Annotation Tool")
    print("=" * 60)
    print(f"Session: {tool.session_dir.name}")
    print(f"Images: {len(tool.image_files)}")
    print()
    print(f"Starting server at http://localhost:{args.port}")
    print("=" * 60)
    
    app.run(host='0.0.0.0', port=args.port, debug=False)


if __name__ == '__main__':
    main()
