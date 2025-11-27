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
                <h3>Legend</h3>
                <div class="legend-item">
                    <div class="legend-color legend-road"></div>
                    <span>🟢 Road (Passable)</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color legend-obstacle"></div>
                    <span>🔴 Obstacle (Avoid)</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color legend-ignorable"></div>
                    <span>▦ Ignorable (Can Touch)</span>
                </div>
            </div>
        </div>
        <div class="right-panel">
            <h2>Detected Objects</h2>
            <div class="object-list" id="object-list"></div>
            <div class="shortcuts">
                <h3>⌨️ Keyboard Shortcuts</h3>
                <div><strong>Space:</strong> Toggle selected object</div>
                <div><strong>→:</strong> Next image</div>
                <div><strong>←:</strong> Previous image</div>
                <div><strong>S:</strong> Save</div>
            </div>
        </div>
    </div>

    <script>
        let currentIndex = 0;
        let images = [];
        let currentObjects = [];
        let selectedObjectId = null;
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
            selectedObjectId = null;
            
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
            
            // Load objects
            currentObjects = data.objects;
            renderObjectList();
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
                    ctx.globalAlpha = 0.5;
                    ctx.drawImage(overlayImg, 0, 0);
                    ctx.globalAlpha = 1.0;
                };
                overlayImg.src = data.overlay_url;
            };
            img.src = data.image_url;
        }

        // Render object list
        function renderObjectList() {
            const list = document.getElementById('object-list');
            list.innerHTML = '';
            
            currentObjects.forEach((obj, idx) => {
                const item = document.createElement('div');
                item.className = 'object-item' + 
                    (obj.enabled ? '' : ' disabled') +
                    (selectedObjectId === obj.id ? ' selected' : '');
                item.onclick = () => selectObject(obj.id);
                
                const info = document.createElement('div');
                info.className = 'object-info';
                
                const name = document.createElement('div');
                name.className = 'object-name';
                name.textContent = obj.label;
                
                const stats = document.createElement('div');
                stats.className = 'object-stats';
                stats.textContent = `${obj.percentage.toFixed(1)}% • ${obj.pixels} px`;
                
                info.appendChild(name);
                info.appendChild(stats);
                
                const toggle = document.createElement('div');
                toggle.className = 'toggle-switch' + (obj.enabled ? ' active' : '');
                toggle.onclick = (e) => {
                    e.stopPropagation();
                    toggleObject(obj.id);
                };
                
                const slider = document.createElement('div');
                slider.className = 'slider';
                toggle.appendChild(slider);
                
                item.appendChild(info);
                item.appendChild(toggle);
                list.appendChild(item);
            });
        }

        // Select object
        function selectObject(id) {
            selectedObjectId = id;
            renderObjectList();
        }

        // Toggle object enabled/disabled
        async function toggleObject(id) {
            const response = await fetch(`/api/toggle/${currentIndex}/${id}`, {
                method: 'POST'
            });
            const data = await response.json();
            currentObjects = data.objects;
            renderObjectList();
            loadImage(currentIndex); // Refresh visualization
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
            if (confirm('Reset all changes for this image?')) {
                const response = await fetch(`/api/reset/${currentIndex}`, {
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
            
            if (data.object_id !== null) {
                selectObject(data.object_id);
            }
        });

        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.key === 'ArrowRight') {
                navigate(1);
            } else if (e.key === 'ArrowLeft') {
                navigate(-1);
            } else if (e.key === ' ' && selectedObjectId !== null) {
                e.preventDefault();
                toggleObject(selectedObjectId);
            } else if (e.key === 's' || e.key === 'S') {
                e.preventDefault();
                save();
            }
        });

        // Initialize
        loadSession();
    </script>
</body>
</html>
"""


class InteractiveAnnotationTool:
    """Interactive web-based annotation tool."""
    
    def __init__(self, session_dir: Path):
        # Resolve session path to an absolute path so file I/O is stable
        self.session_dir = Path(session_dir).resolve()
        self.masks_dir = self.session_dir / 'masks'
        self.images_dir = self.session_dir.parent.parent.parent / 'raw_images' / self.session_dir.name
        self.ade_masks_dir = self.session_dir / 'ade20k_masks'
        self.output_dir = self.session_dir / 'labeled'
        # Ensure the labeled directory exists (create parents if necessary)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load images
        self.image_files = sorted(list(self.images_dir.glob('*.jpg')))
        
        # Load or initialize metadata
        self.metadata_path = self.output_dir / 'metadata.json'
        self.metadata = self._load_metadata()
        
        print(f"Loaded session: {self.session_dir.name}")
        print(f"Found {len(self.image_files)} images")
    
    def _load_metadata(self) -> Dict:
        """Load existing metadata or create new."""
        if self.metadata_path.exists():
            with open(self.metadata_path) as f:
                return json.load(f)
        else:
            # Initialize metadata for all images
            metadata = {}
            for img_file in self.image_files:
                metadata[img_file.stem] = {
                    'objects': {},  # object_id -> enabled
                    'modified': False
                }
            return metadata
    
    def save_metadata(self):
        """Save metadata to disk."""
        with open(self.metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        print(f"Saved metadata: {self.metadata_path}")
    
    def get_objects(self, image_index: int) -> List[Dict]:
        """
        Get detected objects for an image.
        
        Returns list of objects with:
        - id: unique identifier
        - label: ADE20K class name
        - pixels: number of pixels
        - percentage: percentage of image
        - enabled: whether this is a valid obstacle
        """
        img_file = self.image_files[image_index]
        
        # Load ADE20K mask
        ade_mask_path = self.ade_masks_dir / (img_file.stem + '_ade20k.png')
        ade_mask = cv2.imread(str(ade_mask_path), cv2.IMREAD_GRAYSCALE)
        
        # Load JetRacer mask
        jr_mask_path = self.masks_dir / (img_file.stem + '_mask.png')
        jr_mask = cv2.imread(str(jr_mask_path), cv2.IMREAD_GRAYSCALE)
        
        # Get unique ADE IDs (excluding Road which is class 1 in JetRacer)
        objects = []
        for ade_id in np.unique(ade_mask):
            # Only consider pixels that are NOT Road in JetRacer classification
            obj_pixels = (ade_mask == ade_id) & (jr_mask != 1)
            pixel_count = np.sum(obj_pixels)
            
            if pixel_count > 100:  # Minimum size threshold
                obj_id = f"ade_{ade_id}"
                
                # Get enabled status from metadata
                if img_file.stem not in self.metadata:
                    self.metadata[img_file.stem] = {'objects': {}, 'modified': False}
                
                if obj_id not in self.metadata[img_file.stem]['objects']:
                    # Default: enabled (treat as obstacle)
                    self.metadata[img_file.stem]['objects'][obj_id] = True
                
                enabled = self.metadata[img_file.stem]['objects'][obj_id]
                
                objects.append({
                    'id': obj_id,
                    'ade_id': int(ade_id),
                    'label': f"ADE-{ade_id}",  # Could map to actual names
                    'pixels': int(pixel_count),
                    'percentage': float(pixel_count / ade_mask.size * 100),
                    'enabled': enabled
                })
        
        # Sort by pixel count
        objects.sort(key=lambda x: x['pixels'], reverse=True)
        
        return objects
    
    def create_overlay(self, image_index: int) -> np.ndarray:
        """
        Create overlay visualization.
        
        Colors:
        - Green (alpha): Road
        - Red (alpha): Enabled obstacles
        - Yellow stripes: Disabled obstacles (ignorable)
        """
        img_file = self.image_files[image_index]
        
        # Load masks
        jr_mask = cv2.imread(str(self.masks_dir / (img_file.stem + '_mask.png')), cv2.IMREAD_GRAYSCALE)
        ade_mask = cv2.imread(str(self.ade_masks_dir / (img_file.stem + '_ade20k.png')), cv2.IMREAD_GRAYSCALE)
        
        h, w = jr_mask.shape

        # Build ADE20K color map for obstacle visualization (match auto_annotate_decisive)
        ade_color = {}
        for uid in np.unique(ade_mask):
            hval = (int(uid) * 37) % 180
            hsv = np.uint8([[[hval, 200, 200]]])
            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0].tolist()
            ade_color[int(uid)] = tuple(int(x) for x in bgr)

        # Create colored mask (BGR)
        colored_mask = np.zeros((h, w, 3), dtype=np.uint8)
        # Background: gray
        colored_mask[jr_mask == 0] = (128, 128, 128)
        # Road: green
        colored_mask[jr_mask == 1] = (0, 255, 0)
        # Obstacle: color by ADE20K id
        obstacle_idx = (jr_mask == 2)
        for uid, col in ade_color.items():
            colored_mask[np.logical_and(obstacle_idx, ade_mask == uid)] = col

        # Load original image and blend (image is RGB when loaded via PIL earlier; load BGR here)
        img_file = self.image_files[image_index]
        image_bgr = cv2.imread(str(img_file))
        if image_bgr is None:
            # Fallback: return colored mask if image not found
            return colored_mask

        vis_bgr = cv2.addWeighted(image_bgr, 0.5, colored_mask, 0.5, 0)

        # Note: vehicle mask contour drawing is optional; if a vehicle mask exists in session, draw it
        vehicle_mask_path = self.session_dir.parent.parent / 'vehicle_mask.png'
        if vehicle_mask_path.exists():
            vm = cv2.imread(str(vehicle_mask_path), cv2.IMREAD_GRAYSCALE)
            if vm is not None:
                vm_resized = cv2.resize(vm, (vis_bgr.shape[1], vis_bgr.shape[0]))
                contours, _ = cv2.findContours((vm_resized > 127).astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    cv2.drawContours(vis_bgr, contours, -1, (255, 255, 0), 2)

        return vis_bgr


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
    objects = tool.get_objects(index)
    
    return jsonify({
        'image_url': f'/image/{index}',
        'overlay_url': f'/overlay/{index}',
        'objects': objects
    })


@app.route('/image/<int:index>')
def serve_image(index):
    return send_file(tool.image_files[index])


@app.route('/overlay/<int:index>')
def serve_overlay(index):
    # If a precomputed visualization exists (matches auto_annotate output), serve it.
    img_file = tool.image_files[index]
    vis_dir = tool.session_dir.parent / 'visualizations'
    vis_path = vis_dir / (img_file.stem + '_vis.jpg')
    if vis_path.exists():
        return send_file(vis_path)

    # Otherwise, generate overlay on the fly
    overlay = tool.create_overlay(index)

    # Save to temp file
    temp_path = tool.output_dir / f'temp_overlay_{index}.png'
    cv2.imwrite(str(temp_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    return send_file(temp_path)


@app.route('/api/toggle/<int:index>/<object_id>', methods=['POST'])
def api_toggle(index, object_id):
    img_file = tool.image_files[index]
    
    # Toggle
    current = tool.metadata[img_file.stem]['objects'][object_id]
    tool.metadata[img_file.stem]['objects'][object_id] = not current
    tool.metadata[img_file.stem]['modified'] = True
    
    return jsonify({
        'objects': tool.get_objects(index)
    })


@app.route('/api/pick/<int:index>/<int:x>/<int:y>')
def api_pick(index, x, y):
    """Pick object at pixel coordinates."""
    img_file = tool.image_files[index]
    
    # Load ADE mask
    ade_mask_path = tool.ade_masks_dir / (img_file.stem + '_ade20k.png')
    ade_mask = cv2.imread(str(ade_mask_path), cv2.IMREAD_GRAYSCALE)
    
    # Get ADE ID at pixel
    ade_id = int(ade_mask[y, x])
    object_id = f"ade_{ade_id}"
    
    # Check if this object exists
    objects = tool.get_objects(index)
    for obj in objects:
        if obj['id'] == object_id:
            return jsonify({'object_id': object_id})
    
    return jsonify({'object_id': None})


@app.route('/api/save', methods=['POST'])
def api_save():
    tool.save_metadata()
    return jsonify({'message': 'Metadata saved successfully!'})


@app.route('/api/reset/<int:index>', methods=['POST'])
def api_reset(index):
    img_file = tool.image_files[index]
    
    # Reset to default (all enabled)
    tool.metadata[img_file.stem]['objects'] = {}
    tool.metadata[img_file.stem]['modified'] = False
    
    return jsonify({'message': 'Reset successfully'})


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
