"""
Web UI for JetRacer Minimal project.
Phase 1: Includes data collection functionality.
"""
import asyncio
import base64
import cv2
import json
import time
from pathlib import Path
from datetime import datetime
from nicegui import ui
from camera import JetCamera
from segmentation import SegmentationModel
from monitor import PerformanceMonitor


class WebUI:
    """Main Web UI for JetRacer control and data collection."""
    
    def __init__(self, camera, model, monitor, display_config=None):
        """
        Initialize Web UI.
        
        Args:
            camera: JetCamera instance
            model: SegmentationModel instance
            monitor: PerformanceMonitor instance
            display_config: Display configuration dict
        """
        self.camera = camera
        self.model = model
        self.monitor = monitor
        
        # Display configuration
        if display_config is None:
            display_config = {}
        self.target_display_fps = display_config.get('display_fps', 5)
        self.min_send_interval = 1.0 / self.target_display_fps
        self.overlay_enabled = display_config.get('overlay_mask', False)
        self.jpeg_quality = display_config.get('jpeg_quality', 80)
        self.segmentation_interval = display_config.get('segmentation_interval', 5)
        
        # Processing state
        self.running = False
        self.frame_process_count = 0
        self.last_mask = None
        self.last_send_time = 0
        
        # === NEW: Data Collection State ===
        self.data_collection_mode = False
        self.current_session = None
        self.captured_count = 0
        self.session_start_time = None
        self.recent_captures = []  # Store last 5 captures for preview
        
        # === NEW: Dataset Management ===
        self.available_datasets = []
        
        # UI elements (will be initialized in setup_ui)
        self.video_container = None
        self.fps_label = None
        self.status_label = None
        
        # Data collection UI elements
        self.session_label = None
        self.captured_label = None
        self.capture_btn = None
        self.start_session_btn = None
        self.end_session_btn = None
        self.preview_container = None
        self._canvas_script_added = False
    
    def setup_ui(self):
        """Setup complete UI layout."""
        
        with ui.column().classes('w-full items-center gap-4 p-4'):
            # Header
            ui.label('🚗 JetRacer Minimal - Phase 1').classes('text-3xl font-bold')
            
            # Camera feed
            self.setup_camera_display()
            
            # Camera controls
            self.setup_controls()
            
            # === NEW: Data collection panel ===
            self.setup_data_collection_panel()
            
            # Status bar
            self.setup_status_bar()

        # Ensure processing loop starts shortly after UI initializes
        ui.timer(0.1, lambda: self.start_processing(), once=True)
    
    def setup_camera_display(self):
        """Setup camera display area."""
        
        with ui.card().classes('w-full max-w-4xl'):
            ui.label('📹 Camera Feed').classes('text-xl mb-2')
            
            # Video canvas (HTML only, no script)
            self.video_container = ui.html('''
                <canvas id="videoCanvas" width="640" height="480" 
                        style="width: 100%; max-width: 640px; border: 2px solid #ccc; border-radius: 8px;">
                </canvas>
            ''').classes('w-full')
            
            self._inject_canvas_script()

    def _inject_canvas_script(self):
        """Inject canvas update script separately."""
        if self._canvas_script_added:
            return

        ui.add_body_html(
            '''
            <script>
                (function() {
                    function initCanvas() {
                        const canvas = document.getElementById('videoCanvas');
                        if (!canvas) {
                            setTimeout(initCanvas, 100);
                            return;
                        }

                        const ctx = canvas.getContext('2d');
                        window.updateVideoFrame = function(base64Data) {
                            const img = new Image();
                            img.onload = function() {
                                ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
                            };
                            img.src = 'data:image/jpeg;base64,' + base64Data;
                        };

                        console.log('[Canvas] Update function initialized');
                    }

                    if (document.readyState === 'loading') {
                        document.addEventListener('DOMContentLoaded', initCanvas);
                    } else {
                        initCanvas();
                    }
                })();
            </script>
            '''
        )
        self._canvas_script_added = True
    
    def setup_controls(self):
        """Setup camera control panel."""
        
        with ui.card().classes('w-full max-w-4xl'):
            ui.label('🎛️ Camera Controls').classes('text-xl mb-2')
            
            with ui.row().classes('gap-4 items-center'):
                # Display FPS control
                with ui.column():
                    ui.label('Display FPS:').classes('text-sm')
                    ui.slider(
                        min=5, max=20, step=1, value=self.target_display_fps,
                        on_change=lambda e: setattr(self, 'target_display_fps', int(e.value))
                    ).props('label-always')
                
                # Overlay toggle
                with ui.column():
                    ui.label('Segmentation Overlay:').classes('text-sm')
                    ui.switch(
                        'Enable',
                        value=self.overlay_enabled,
                        on_change=lambda e: setattr(self, 'overlay_enabled', e.value)
                    )
                
                # JPEG quality
                with ui.column():
                    ui.label('JPEG Quality:').classes('text-sm')
                    ui.slider(
                        min=70, max=95, step=5, value=self.jpeg_quality,
                        on_change=lambda e: setattr(self, 'jpeg_quality', int(e.value))
                    ).props('label-always')
    
    def setup_data_collection_panel(self):
        """
        Setup data collection panel for capturing training images.
        
        Features:
        - Start/end session management
        - Single-click image capture
        - Real-time counter
        - Recent captures preview
        """
        
        with ui.card().classes('w-full max-w-4xl mt-4'):
            ui.label('📸 Data Collection').classes('text-2xl font-bold mb-4')
            
            # Session Info Row
            with ui.row().classes('gap-4 items-center mb-4'):
                with ui.column():
                    ui.label('Current Session:').classes('text-sm text-gray-600')
                    self.session_label = ui.label('None').classes('text-lg font-bold text-blue-600')
                
                with ui.column():
                    ui.label('Frames Captured:').classes('text-sm text-gray-600')
                    self.captured_label = ui.label('0').classes('text-2xl font-bold text-green-600')
            
            # Control Buttons Row
            with ui.row().classes('gap-4 mt-4'):
                # Start New Session Button
                self.start_session_btn = ui.button(
                    '🎬 Start New Session',
                    on_click=self.start_collection_session
                ).props('color=blue size=lg').classes('px-6 py-3')
                
                # Capture Button (Large and prominent)
                self.capture_btn = ui.button(
                    '📷 CAPTURE',
                    on_click=self.capture_frame
                ).props('color=green size=xl').classes('text-3xl px-12 py-6 font-bold')
                self.capture_btn.set_enabled(False)
                
                # End Session Button
                self.end_session_btn = ui.button(
                    '⏹️ End Session',
                    on_click=self.end_collection_session
                ).props('color=orange size=lg').classes('px-6 py-3')
                self.end_session_btn.set_enabled(False)
            
            # Instructions
            with ui.expansion('📖 Instructions', icon='help').classes('w-full mt-4'):
                ui.markdown("""
                **How to use Data Collection:**
                
                1. Click **Start New Session** to begin
                2. Click **CAPTURE** button repeatedly to save frames
                3. Aim for 50-100 frames per session
                4. Click **End Session** when done
                5. Frames saved to `data/raw_images/session_YYYYMMDD_HHMMSS/`
                6. Use Label Studio for annotation
                """)
            
            # Recent Captures Preview
            with ui.card().tight().classes('w-full mt-4 p-2'):
                ui.label('Recent Captures:').classes('text-sm text-gray-600 mb-2')
                self.preview_container = ui.row().classes('gap-2')
    
    def setup_status_bar(self):
        """Setup status bar."""
        
        with ui.card().classes('w-full max-w-4xl'):
            with ui.row().classes('gap-4 items-center'):
                ui.label('Status:').classes('text-sm')
                self.status_label = ui.label('Ready').classes('font-bold')
                
                ui.label('FPS:').classes('text-sm ml-4')
                self.fps_label = ui.label('0.0').classes('font-bold')
    
    # =========================================================================
    # Data Collection Methods
    # =========================================================================
    
    def start_collection_session(self):
        """
        Start a new data collection session.
        Creates timestamped directory and initializes metadata.
        """
        # Generate session ID
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.current_session = f"session_{timestamp}"
        self.session_start_time = datetime.now().isoformat()
        
        # Create session directory
        session_path = Path(f"data/raw_images/{self.current_session}")
        session_path.mkdir(parents=True, exist_ok=True)
        
        # Reset counter
        self.captured_count = 0
        self.recent_captures = []
        
        # Update UI
        self.session_label.set_text(self.current_session)
        self.captured_label.set_text('0')
        self.capture_btn.set_enabled(True)
        self.end_session_btn.set_enabled(True)
        self.start_session_btn.set_enabled(False)
        
        # Log
        print(f"[DataCollection] Started session: {self.current_session}")
        ui.notify(f'✓ Session started: {self.current_session}', type='positive')
    
    def capture_frame(self):
        """
        Capture current camera frame and save to disk.
        Updates counter and preview.
        """
        if not self.current_session:
            ui.notify('⚠️ Start a session first!', type='warning')
            return
        
        # Get current frame from camera
        frame = self.camera.read()
        if frame is None:
            ui.notify('❌ Failed to capture frame', type='negative')
            return
        
        # Increment counter
        self.captured_count += 1
        
        # Generate filename
        filename = f"img_{self.captured_count:04d}.jpg"
        filepath = Path(f"data/raw_images/{self.current_session}/{filename}")
        
        # Save frame (high quality for annotation)
        cv2.imwrite(str(filepath), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        
        # Save frame metadata (optional but useful)
        meta_filepath = filepath.with_suffix('.json')
        metadata = {
            'filename': filename,
            'timestamp': datetime.now().isoformat(),
            'frame_number': self.captured_count,
            'camera_settings': {
                'width': self.camera.width,
                'height': self.camera.height,
                'fps': self.camera.fps
            }
        }
        with open(meta_filepath, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Update UI
        self.captured_label.set_text(str(self.captured_count))
        
        # Update preview (keep last 5)
        self.recent_captures.append(str(filepath))
        if len(self.recent_captures) > 5:
            self.recent_captures.pop(0)
        
        # Log
        print(f"[DataCollection] Captured: {filename}")
        
        # Brief notification (don't spam)
        if self.captured_count % 10 == 0:
            ui.notify(f'✓ {self.captured_count} frames captured', type='info')
    
    def end_collection_session(self):
        """
        End current data collection session.
        Saves metadata and shows next steps.
        """
        if not self.current_session:
            return
        
        # Calculate session duration
        end_time = datetime.now()
        
        # Save session metadata
        metadata = {
            'session_id': self.current_session,
            'start_time': self.session_start_time,
            'end_time': end_time.isoformat(),
            'total_frames': self.captured_count,
            'camera_settings': {
                'width': self.camera.width,
                'height': self.camera.height,
                'fps': self.camera.fps
            }
        }
        
        metadata_path = Path(f"data/raw_images/{self.current_session}/metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Update UI
        self.capture_btn.set_enabled(False)
        self.end_session_btn.set_enabled(False)
        self.start_session_btn.set_enabled(True)
        
        # Show summary
        summary = f"✓ Session complete: {self.captured_count} frames saved"
        print(f"[DataCollection] {summary}")
        ui.notify(summary, type='positive')
        
        # Show next steps dialog
        self.show_annotation_instructions()
    
    def show_annotation_instructions(self):
        """
        Display instructions for annotation workflow.
        """
        session_path = f"data/raw_images/{self.current_session}"
        
        instructions = f"""
## ✅ Data Collection Complete!

**Captured**: {self.captured_count} frames  
**Location**: `{session_path}`

### Next Steps:

1. **Start Label Studio**:
   ```bash
   label-studio start --port 8081 --data-dir ./data/label_studio
   ```

2. **Open in browser**: http://localhost:8081

3. **Import images** from: `{session_path}`

4. **Annotate** using polygon tool:
   - 🟢 Green: Floor (drivable area)
   - 🔴 Red: Obstacle
   - 🔵 Blue: Line (white line, lane marking)

5. **Export annotations** (JSON format)

6. **Convert to masks**:
   ```bash
   python3 scripts/convert_annotations.py \\
     --input data/annotations/export/annotations.json \\
     --output data/datasets/v1
   ```

7. **Return here** and start training!
        """
        
        with ui.dialog() as dialog, ui.card().classes('max-w-3xl'):
            ui.markdown(instructions).classes('text-sm')
            ui.button('OK', on_click=dialog.close).props('color=green')
        
        dialog.open()
    
    # =========================================================================
    # Processing Loop
    # =========================================================================
    
    async def process_loop(self):
        """Main processing loop for video streaming."""
        
        print("[WebUI] Starting process loop...")
        self.running = True
        self.status_label.set_text('Running')
        
        while self.running:
            try:
                # Read frame from camera
                frame = self.camera.read()
                if frame is None:
                    await asyncio.sleep(0.01)
                    continue
                
                self.frame_process_count += 1
                
                # Run segmentation (conditionally)
                if self.overlay_enabled and (self.frame_process_count % self.segmentation_interval == 0):
                    start_time = time.time()
                    self.last_mask = self.model.inference(frame)
                    inference_time = time.time() - start_time
                    self.monitor.update(inference_time)
                else:
                    self.monitor.update(0.0)
                
                # Apply overlay if enabled
                if self.overlay_enabled and self.last_mask is not None:
                    frame = self.apply_overlay(frame, self.last_mask)
                
                # Send frame to browser (rate limited)
                current_time = time.time()
                if current_time - self.last_send_time >= self.min_send_interval:
                    await self.send_frame(frame)
                    self.last_send_time = current_time
                
                # Update FPS display
                if self.frame_process_count % 10 == 0:
                    fps = self.monitor.get_fps()
                    self.fps_label.set_text(f'{fps:.1f}')
                
                # Small delay to prevent CPU saturation
                await asyncio.sleep(0.001)
                
            except Exception as e:
                print(f"[WebUI] Error in process loop: {e}")
                await asyncio.sleep(0.1)
        
        print("[WebUI] Process loop stopped")
        self.status_label.set_text('Stopped')
    
    def apply_overlay(self, frame, mask):
        """
        Apply segmentation overlay to frame.
        
        Args:
            frame: BGR image
            mask: Binary mask
            
        Returns:
            Frame with overlay
        """
        overlay = frame.copy()
        overlay[mask > 0] = [0, 255, 0]  # Green for drivable area
        return cv2.addWeighted(frame, 0.6, overlay, 0.4, 0)
    
    async def send_frame(self, frame):
        """
        Send frame to browser via JavaScript.
        
        Args:
            frame: BGR image
        """
        try:
            # Encode to JPEG
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
            
            # Convert to base64
            img_base64 = base64.b64encode(buffer.tobytes()).decode('ascii')
            payload = json.dumps(img_base64)

            script = f'window.updateVideoFrame({payload})'
            self.video_container.client.run_javascript(script)
        except asyncio.TimeoutError:
            print("[WebUI] Warning: Frame send timeout")
        except Exception as e:
            print(f"[WebUI] Error sending frame: {e}")
    
    def start_processing(self):
        """Start processing loop."""
        if not self.running:
            asyncio.create_task(self.process_loop())
    
    def stop_processing(self):
        """Stop processing loop."""
        self.running = False
