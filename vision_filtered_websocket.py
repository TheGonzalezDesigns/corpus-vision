import cv2
import base64
import time
import threading
import queue
import json
import logging
import requests
from flask import Flask
from flask_socketio import SocketIO, emit
from corpus_vision import VisionSystem

# Import the Rust filter (will be built with maturin)
try:
    from frame_change_detector import FrameChangeDetector
    FILTER_AVAILABLE = True
except ImportError:
    logging.warning("Rust filter not available. Install with: cd filters/frame-change-detector && maturin develop")
    FILTER_AVAILABLE = False

app = Flask(__name__)
app.config['SECRET_KEY'] = 'corpus-vision-filtered-websocket'
socketio = SocketIO(app, cors_allowed_origins="*", ping_timeout=60, ping_interval=25)

class FilteredVisionWebSocketServer:
    def __init__(self):
        self.vision = None
        self.filter = None
        self.streaming = False
        self.capture_thread = None
        self.frame_queue = queue.Queue(maxsize=50)
        self.client_count = 0
        self.config = {
            'frame_interval_ms': 33,     # 30fps capture (1000ms/30fps = 33ms)
            'buffer_duration_ms': 100,   # 100ms comparison window  
            'change_threshold': 5.0,     # 5% change to trigger AI
            'filter_enabled': True,      # Use filter or process all frames
            'resolution': {'width': 3840, 'height': 2160}  # 4K resolution
        }
        self.stats = {
            'frames_captured': 0,
            'frames_processed': 0,
            'frames_triggered': 0,
            'start_time': None,
            'ai_calls_saved': 0
        }
        
    def initialize_systems(self):
        """Initialize vision system and filter"""
        try:
            self.vision = VisionSystem()
            
            # Configure camera for 4K resolution
            if self.vision.camera:
                self.vision.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.config['resolution']['width'])
                self.vision.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config['resolution']['height'])
                self.vision.camera.set(cv2.CAP_PROP_FPS, 30)
                logging.info("Camera configured for 4K @ 30fps")
            
            logging.info("Vision system initialized for filtered WebSocket")
            
            if FILTER_AVAILABLE:
                self.filter = FrameChangeDetector(
                    buffer_duration_ms=self.config['buffer_duration_ms'],
                    change_threshold=self.config['change_threshold'],
                    frame_interval_ms=self.config['frame_interval_ms']
                )
                logging.info("Rust frame change filter initialized")
            else:
                logging.warning("Filter not available - will process all frames")
            
            return True
        except Exception as e:
            logging.error(f"Failed to initialize systems: {e}")
            return False
    
    def start_filtered_stream(self):
        """Start filtered frame capture with AI triggering"""
        if self.streaming:
            logging.warning("Filtered stream already running")
            return False
        
        if not self.vision:
            if not self.initialize_systems():
                return False
        
        self.streaming = True
        self.stats['start_time'] = time.time()
        self._reset_stats()
        
        def filtered_capture_worker():
            interval = self.config['frame_interval_ms'] / 1000.0
            last_capture = 0
            
            while self.streaming:
                current_time = time.time()
                timestamp_ms = int(current_time * 1000)
                
                if current_time - last_capture >= interval:
                    try:
                        # Capture frame
                        frame = self.vision.capture_image()
                        if frame is not None:
                            self.stats['frames_captured'] += 1
                            
                            # Convert to base64
                            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                            frame_b64 = base64.b64encode(buffer).decode('utf-8')
                            
                            # Process through filter
                            should_trigger, change_pct, buffer_size = self._process_with_filter(
                                frame_b64, timestamp_ms
                            )
                            
                            self.stats['frames_processed'] += 1
                            
                            # Send frame data to clients
                            frame_data = {
                                'timestamp': timestamp_ms,
                                'frame': frame_b64,
                                'change_detected': should_trigger,
                                'change_percentage': change_pct,
                                'buffer_size': buffer_size,
                                'frame_number': self.stats['frames_captured']
                            }
                            
                            # Add to queue
                            try:
                                self.frame_queue.put(frame_data, block=False)
                            except queue.Full:
                                pass
                            
                            # Trigger AI analysis if filter says so
                            if should_trigger:
                                self.stats['frames_triggered'] += 1
                                self._trigger_ai_analysis(frame_b64, timestamp_ms)
                            else:
                                self.stats['ai_calls_saved'] += 1
                        
                        last_capture = current_time
                    except Exception as e:
                        logging.error(f"Filtered capture error: {e}")
                
                time.sleep(0.001)
            
            logging.info("Filtered capture thread stopped")
        
        self.capture_thread = threading.Thread(target=filtered_capture_worker, daemon=True)
        self.capture_thread.start()
        logging.info(f"Started filtered stream at {1000/self.config['frame_interval_ms']:.1f} fps")
        return True
    
    def _process_with_filter(self, frame_b64: str, timestamp_ms: int) -> tuple:
        """Process frame through Rust filter"""
        if self.filter and self.config['filter_enabled']:
            try:
                return self.filter.process_frame(frame_b64, timestamp_ms)
            except Exception as e:
                logging.error(f"Filter processing error: {e}")
                return (True, 100.0, 0)  # Fail-safe: trigger AI if filter fails
        else:
            # No filter - always trigger (for testing)
            return (True, 100.0, 0)
    
    def _trigger_ai_analysis(self, frame_b64: str, timestamp_ms: int):
        """Trigger AI analysis and speech for significant changes"""
        def ai_worker():
            try:
                # Decode frame for AI analysis
                img_data = base64.b64decode(frame_b64)
                nparr = np.frombuffer(img_data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if frame is not None:
                    # Get AI description
                    description = self.vision.analyze_image(frame)
                    if description:
                        # Send to speech API
                        self._send_to_speech(description, timestamp_ms)
                        
                        # Notify clients about AI trigger
                        socketio.emit('ai_triggered', {
                            'timestamp': timestamp_ms,
                            'description': description,
                            'trigger_reason': 'significant_change_detected'
                        })
            except Exception as e:
                logging.error(f"AI analysis error: {e}")
        
        # Run AI analysis in separate thread to avoid blocking
        threading.Thread(target=ai_worker, daemon=True).start()
    
    def _send_to_speech(self, description: str, timestamp_ms: int):
        """Send description to speech API"""
        try:
            speech_url = self.vision.config['speech']['speech_api_url']
            response = requests.post(
                f"{speech_url}/speak",
                json={'text': description},
                timeout=5
            )
            logging.info(f"Speech triggered: {description[:50]}...")
        except Exception as e:
            logging.error(f"Speech API error: {e}")
    
    def stop_stream(self):
        """Stop filtered streaming"""
        self.streaming = False
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=2)
        logging.info("Filtered stream stopped")
    
    def get_latest_frame(self):
        """Get latest frame from queue"""
        try:
            return self.frame_queue.get_nowait()
        except queue.Empty:
            return None
    
    def _reset_stats(self):
        """Reset statistics"""
        for key in ['frames_captured', 'frames_processed', 'frames_triggered', 'ai_calls_saved']:
            self.stats[key] = 0
    
    def get_performance_stats(self):
        """Get performance metrics"""
        if self.stats['start_time']:
            uptime = time.time() - self.stats['start_time']
            capture_fps = self.stats['frames_captured'] / uptime if uptime > 0 else 0
            trigger_rate = (self.stats['frames_triggered'] / self.stats['frames_processed'] * 100) if self.stats['frames_processed'] > 0 else 0
            
            return {
                'uptime_seconds': uptime,
                'capture_fps': round(capture_fps, 1),
                'frames_captured': self.stats['frames_captured'],
                'frames_processed': self.stats['frames_processed'],
                'ai_triggers': self.stats['frames_triggered'],
                'ai_calls_saved': self.stats['ai_calls_saved'],
                'trigger_rate_percent': round(trigger_rate, 1),
                'filter_enabled': self.config['filter_enabled'],
                'change_threshold': self.config['change_threshold']
            }
        return {}

# Global instance
filtered_vision = FilteredVisionWebSocketServer()

@socketio.on('connect')
def handle_connect():
    filtered_vision.client_count += 1
    logging.info(f"Client connected to filtered vision. Total: {filtered_vision.client_count}")
    emit('status', {
        'connected': True, 
        'message': 'Connected to filtered vision WebSocket',
        'filter_available': FILTER_AVAILABLE,
        'config': filtered_vision.config
    })

@socketio.on('disconnect')
def handle_disconnect():
    filtered_vision.client_count -= 1
    logging.info(f"Client disconnected. Total: {filtered_vision.client_count}")
    
    if filtered_vision.client_count <= 0:
        filtered_vision.stop_stream()

@socketio.on('start_filtered_stream')
def handle_start_filtered_stream(data=None):
    """Start intelligent filtered streaming"""
    if data:
        # Update configuration
        for key in ['frame_interval_ms', 'change_threshold', 'filter_enabled']:
            if key in data:
                filtered_vision.config[key] = data[key]
    
    if filtered_vision.start_filtered_stream():
        emit('stream_started', {
            'status': 'success',
            'message': 'Filtered vision stream started',
            'config': filtered_vision.config,
            'filter_available': FILTER_AVAILABLE
        })
        
        # Start frame sender
        def filtered_frame_sender():
            while filtered_vision.streaming and filtered_vision.client_count > 0:
                frame_data = filtered_vision.get_latest_frame()
                if frame_data:
                    socketio.emit('filtered_frame', frame_data)
                time.sleep(0.02)  # 50fps max send rate
        
        threading.Thread(target=filtered_frame_sender, daemon=True).start()
    else:
        emit('error', {'message': 'Failed to start filtered stream'})

@socketio.on('stop_stream')
def handle_stop_stream():
    """Stop filtered streaming"""
    filtered_vision.stop_stream()
    emit('stream_stopped', {'status': 'success', 'message': 'Filtered stream stopped'})

@socketio.on('configure_filter')
def handle_configure_filter(data):
    """Configure filter parameters"""
    old_config = filtered_vision.config.copy()
    
    # Update configuration
    for key in ['change_threshold', 'filter_enabled', 'frame_interval_ms', 'buffer_duration_ms']:
        if key in data:
            filtered_vision.config[key] = data[key]
    
    # Update filter if available
    if filtered_vision.filter and FILTER_AVAILABLE:
        try:
            filtered_vision.filter.configure(
                buffer_duration_ms=filtered_vision.config['buffer_duration_ms'],
                change_threshold=filtered_vision.config['change_threshold'], 
                frame_interval_ms=filtered_vision.config['frame_interval_ms']
            )
        except Exception as e:
            logging.error(f"Filter configuration error: {e}")
    
    emit('filter_configured', {
        'status': 'success',
        'old_config': old_config,
        'new_config': filtered_vision.config
    })

@socketio.on('get_performance')
def handle_get_performance():
    """Get performance statistics"""
    stats = filtered_vision.get_performance_stats()
    emit('performance_stats', stats)

if __name__ == '__main__':
    import numpy as np
    logging.basicConfig(level=logging.INFO)
    logging.info("Starting Filtered Vision WebSocket Server on port 5002")
    logging.info(f"Rust filter available: {FILTER_AVAILABLE}")
    socketio.run(app, host='0.0.0.0', port=5002, debug=False)