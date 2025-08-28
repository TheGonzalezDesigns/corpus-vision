import cv2
import base64
import time
import threading
import queue
import json
import logging
from flask import Flask
from flask_socketio import SocketIO, emit
from corpus_vision import VisionSystem

app = Flask(__name__)
app.config['SECRET_KEY'] = 'corpus-vision-websocket'
socketio = SocketIO(app, cors_allowed_origins="*", ping_timeout=60, ping_interval=25)

class VisionWebSocketServer:
    def __init__(self):
        self.vision = None
        self.streaming = False
        self.capture_thread = None
        self.frame_queue = queue.Queue(maxsize=10)
        self.client_count = 0
        self.stats = {
            'frames_captured': 0,
            'frames_sent': 0,
            'start_time': None,
            'fps': 0
        }
        
    def initialize_vision(self):
        """Initialize vision system"""
        try:
            self.vision = VisionSystem()
            logging.info("Vision system initialized for WebSocket")
            return True
        except Exception as e:
            logging.error(f"Failed to initialize vision system: {e}")
            return False
    
    def start_frame_capture(self, interval_ms=20):
        """Start capturing frames at specified interval (default 20ms = 50fps)"""
        if self.streaming:
            logging.warning("Frame capture already running")
            return False
        
        if not self.vision:
            if not self.initialize_vision():
                return False
        
        self.streaming = True
        self.stats['start_time'] = time.time()
        self.stats['frames_captured'] = 0
        self.stats['frames_sent'] = 0
        
        def capture_worker():
            interval = interval_ms / 1000.0  # Convert to seconds
            last_capture = 0
            
            while self.streaming:
                current_time = time.time()
                
                # Maintain consistent frame rate
                if current_time - last_capture >= interval:
                    try:
                        frame = self.vision.capture_image()
                        if frame is not None:
                            # Convert frame to base64
                            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                            frame_b64 = base64.b64encode(buffer).decode('utf-8')
                            
                            # Add to queue (non-blocking)
                            try:
                                self.frame_queue.put({
                                    'timestamp': current_time,
                                    'frame': frame_b64,
                                    'frame_number': self.stats['frames_captured']
                                }, block=False)
                                
                                self.stats['frames_captured'] += 1
                            except queue.Full:
                                # Skip frame if queue is full
                                pass
                        
                        last_capture = current_time
                    except Exception as e:
                        logging.error(f"Frame capture error: {e}")
                
                time.sleep(0.001)  # Small sleep to prevent CPU spinning
            
            logging.info("Frame capture thread stopped")
        
        self.capture_thread = threading.Thread(target=capture_worker, daemon=True)
        self.capture_thread.start()
        logging.info(f"Started frame capture at {1000/interval_ms:.1f} fps")
        return True
    
    def stop_frame_capture(self):
        """Stop frame capture"""
        self.streaming = False
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=2)
        logging.info("Frame capture stopped")
    
    def get_latest_frame(self):
        """Get latest frame from queue"""
        try:
            return self.frame_queue.get_nowait()
        except queue.Empty:
            return None
    
    def calculate_fps(self):
        """Calculate actual FPS"""
        if self.stats['start_time'] and self.stats['frames_captured'] > 0:
            elapsed = time.time() - self.stats['start_time']
            return self.stats['frames_captured'] / elapsed if elapsed > 0 else 0
        return 0

# Global instance
vision_ws = VisionWebSocketServer()

@socketio.on('connect')
def handle_connect():
    vision_ws.client_count += 1
    logging.info(f"Client connected. Total clients: {vision_ws.client_count}")
    emit('status', {'connected': True, 'message': 'Connected to vision WebSocket'})

@socketio.on('disconnect') 
def handle_disconnect():
    vision_ws.client_count -= 1
    logging.info(f"Client disconnected. Total clients: {vision_ws.client_count}")
    
    # Stop streaming if no clients
    if vision_ws.client_count <= 0:
        vision_ws.stop_frame_capture()

@socketio.on('start_stream')
def handle_start_stream(data=None):
    """Start real-time frame streaming"""
    config = data or {}
    interval_ms = config.get('interval_ms', 20)  # Default 20ms = 50fps
    
    if vision_ws.start_frame_capture(interval_ms):
        emit('stream_started', {
            'status': 'success', 
            'message': f'Frame capture started at {1000/interval_ms:.1f} fps',
            'interval_ms': interval_ms
        })
        
        # Start frame sender
        def frame_sender():
            while vision_ws.streaming and vision_ws.client_count > 0:
                frame_data = vision_ws.get_latest_frame()
                if frame_data:
                    frame_data['fps'] = vision_ws.calculate_fps()
                    socketio.emit('frame', frame_data)
                    vision_ws.stats['frames_sent'] += 1
                time.sleep(0.01)  # 100fps max send rate
        
        threading.Thread(target=frame_sender, daemon=True).start()
    else:
        emit('error', {'message': 'Failed to start frame capture'})

@socketio.on('stop_stream')
def handle_stop_stream():
    """Stop frame streaming"""
    vision_ws.stop_frame_capture()
    emit('stream_stopped', {'status': 'success', 'message': 'Frame streaming stopped'})

@socketio.on('get_stats')
def handle_get_stats():
    """Get streaming statistics"""
    fps = vision_ws.calculate_fps()
    vision_ws.stats['fps'] = fps
    
    emit('stats', {
        'streaming': vision_ws.streaming,
        'clients': vision_ws.client_count,
        'frames_captured': vision_ws.stats['frames_captured'],
        'frames_sent': vision_ws.stats['frames_sent'],
        'fps': fps,
        'uptime': time.time() - vision_ws.stats['start_time'] if vision_ws.stats['start_time'] else 0
    })

@socketio.on('configure')
def handle_configure(data):
    """Configure streaming parameters"""
    # This will be used to configure the filter
    interval_ms = data.get('interval_ms', 20)
    filter_enabled = data.get('filter_enabled', True)
    change_threshold = data.get('change_threshold', 5.0)  # Percentage
    
    emit('configured', {
        'status': 'success',
        'config': {
            'interval_ms': interval_ms,
            'filter_enabled': filter_enabled,
            'change_threshold': change_threshold
        }
    })

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logging.info("Starting Vision WebSocket Server on port 5002")
    socketio.run(app, host='0.0.0.0', port=5002, debug=False)