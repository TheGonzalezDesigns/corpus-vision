from flask import Flask, request
from flask_restx import Api, Resource, fields
from flask_socketio import SocketIO, emit
import logging
import yaml
import cv2
import numpy as np
import base64
import time
import threading
import queue
from corpus_vision import VisionSystem
from waldo_vision_logger import waldo_logger
from continuous_waldo_monitor import waldo_monitor

# Import Rust filter
try:
    from frame_change_detector import FrameChangeDetector
    FILTER_AVAILABLE = True
    waldo_logger.logger.info("🦀 Waldo Vision filter available")
except ImportError:
    logging.warning("Rust filter not available. Build with: cd ../../filters/frame-change-detector && ./build.sh")
    FILTER_AVAILABLE = False
    waldo_logger.logger.info("❌ Waldo Vision filter not available")

app = Flask(__name__)
app.config['SECRET_KEY'] = 'corpus-vision-api'

# Initialize SocketIO for WebSocket support
socketio = SocketIO(app, cors_allowed_origins="*", ping_timeout=60, ping_interval=25)

api = Api(app,
    version='1.0', 
    title='Corpus Vision API',
    description='Computer vision and image analysis with WebSocket streaming and intelligent filtering',
    doc='/swagger'
)

logging.basicConfig(level=logging.INFO)

# Initialize Vision System
try:
    vision = VisionSystem()
    
    # Configure for 4K @ 30fps
    if vision.camera:
        vision.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
        vision.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)  
        vision.camera.set(cv2.CAP_PROP_FPS, 30)
        logging.info("Camera configured for 4K @ 30fps")
        
except Exception as e:
    logging.error(f"Failed to initialize vision system: {e}")
    vision = None

# Initialize Streaming System
class StreamingManager:
    def __init__(self):
        self.streaming = False
        self.filter = None
        self.frame_queue = queue.Queue(maxsize=20)
        self.client_count = 0
        self.config = {
            'frame_interval_ms': 33,    # 30fps
            'change_threshold': 5.0,    # 5% change
            'filter_enabled': True
        }
        self.stats = {'frames': 0, 'triggers': 0, 'saved': 0, 'start': None}
    
    def initialize_filter(self):
        if FILTER_AVAILABLE:
            self.filter = FrameChangeDetector(
                100,  # buffer_duration_ms
                self.config['change_threshold'],  # change_threshold
                self.config['frame_interval_ms']  # frame_interval_ms
            )
            return True
        return False

streaming = StreamingManager()

# Define API models
loop_model = api.model('LoopRequest', {
    'interval': fields.Integer(description='Seconds between descriptions', example=5, default=5)
})

config_model = api.model('ConfigRequest', {
    'interval': fields.Integer(description='Seconds between descriptions', example=5),
    'first_person': fields.Boolean(description='Use first-person perspective', example=True),
    'speech_enabled': fields.Boolean(description='Enable speech output', example=True)
})

# Quality preset combinations (based on actual Logitech BRIO capabilities)
QUALITY_PRESETS = {
    'high_quality_1080p': {'width': 1920, 'height': 1080, 'fps': 5, 'interval_ms': 200, 'description': '1080p @ 5fps (High Quality)'},
    'balanced_720p': {'width': 1280, 'height': 720, 'fps': 10, 'interval_ms': 100, 'description': '720p @ 10fps (Balanced)'},
    'smooth_480p': {'width': 640, 'height': 480, 'fps': 30, 'interval_ms': 33, 'description': '480p @ 30fps (Smooth Motion)'},
    'legacy_1080p': {'width': 1920, 'height': 1080, 'fps': 5, 'interval_ms': 200, 'description': '1080p @ 5fps (Legacy)'},
    'legacy_720p': {'width': 1280, 'height': 720, 'fps': 10, 'interval_ms': 100, 'description': '720p @ 10fps (Legacy)'}
}

stream_config_model = api.model('StreamConfigRequest', {
    'quality_preset': fields.String(description='Quality and frame rate preset', 
                                   enum=list(QUALITY_PRESETS.keys()), example='smooth_480p'),
    'frame_interval_ms': fields.Integer(description='Custom milliseconds between frames', example=33, enum=[33, 50, 100, 150, 200, 250, 500, 1000]),
    'change_threshold': fields.Float(description='Percentage change to trigger AI analysis', example=5.0, enum=[1.0, 2.5, 5.0, 7.5, 10.0, 15.0, 20.0]),
    'filter_enabled': fields.Boolean(description='Enable intelligent change detection filter', example=True),
    'buffer_duration_ms': fields.Integer(description='Frame comparison window duration', example=100, enum=[50, 100, 150, 200])
})

camera_config_model = api.model('CameraConfigRequest', {
    'quality_preset': fields.String(description='Camera quality preset', 
                                   enum=list(QUALITY_PRESETS.keys()), example='smooth_480p'),
    'width': fields.Integer(description='Custom width (overrides preset)', example=640),
    'height': fields.Integer(description='Custom height (overrides preset)', example=480),
    'fps': fields.Integer(description='Custom FPS (overrides preset)', example=30)
})

stream_control_model = api.model('StreamControlRequest', {
    'action': fields.String(required=True, description='Stream control action', enum=['start', 'stop'], example='start')
})

success_response = api.model('SuccessResponse', {
    'status': fields.String(description='Response status'),
    'message': fields.String(description='Response message')
})

error_response = api.model('ErrorResponse', {
    'error': fields.String(description='Error message')
})

status_response = api.model('StatusResponse', {
    'status': fields.String(description='Service status'),
    'module': fields.String(description='Module name'),
    'camera_available': fields.Boolean(description='Camera availability'),
    'gemini_available': fields.Boolean(description='Gemini AI availability'),
    'continuous_running': fields.Boolean(description='Continuous vision status'),
    'speech_enabled': fields.Boolean(description='Speech integration status')
})

capture_response = api.model('CaptureResponse', {
    'status': fields.String(description='Response status'),
    'image': fields.String(description='Base64 encoded image'),
    'timestamp': fields.Integer(description='Capture timestamp')
})

analyze_response = api.model('AnalyzeResponse', {
    'status': fields.String(description='Response status'),
    'description': fields.String(description='AI-generated description'),
    'timestamp': fields.Integer(description='Analysis timestamp')
})

describe_response = api.model('DescribeResponse', {
    'status': fields.String(description='Response status'),
    'description': fields.String(description='AI-generated description'),
    'spoken': fields.Boolean(description='Whether description was spoken')
})

@api.route('/status')
class Status(Resource):
    @api.response(200, 'Success', status_response)
    @api.response(500, 'Internal Server Error', error_response)
    def get(self):
        """Get vision system status and capabilities"""
        if not vision:
            return {"error": "Vision system not initialized"}, 500
        
        return {
            "status": "running",
            "module": "corpus-vision",
            **vision.get_status()
        }

@api.route('/capture')
class Capture(Resource):
    @api.response(200, 'Success', capture_response)
    @api.response(500, 'Internal Server Error', error_response)
    def get(self):
        """Capture current camera image as base64"""
        if not vision:
            return {"error": "Vision system not initialized"}, 500
        
        image = vision.capture_image()
        if image is None:
            return {"error": "Failed to capture image"}, 500
        
        # Convert image to base64 for JSON response
        _, buffer = cv2.imencode('.jpg', image)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return {
            "status": "success",
            "image": f"data:image/jpeg;base64,{img_base64}",
            "timestamp": int(time.time())
        }

@api.route('/analyze')
class Analyze(Resource):
    @api.response(200, 'Success', analyze_response)
    @api.response(500, 'Internal Server Error', error_response)
    def post(self):
        """Analyze current camera view with AI"""
        if not vision:
            return {"error": "Vision system not initialized"}, 500
        
        image = vision.capture_image()
        if image is None:
            return {"error": "Failed to capture image"}, 500
        
        description = vision.analyze_image(image)
        if description is None:
            return {"error": "Failed to analyze image"}, 500
        
        return {
            "status": "success",
            "description": description,
            "timestamp": int(time.time())
        }

@api.route('/describe')
class Describe(Resource):
    @api.response(200, 'Success', describe_response)
    @api.response(500, 'Internal Server Error', error_response)
    def get(self):
        """Get AI description of current view (includes speech if enabled)"""
        if not vision:
            return {"error": "Vision system not initialized"}, 500
        
        description = vision.get_current_view_description()
        if description is None:
            return {"error": "Failed to get description"}, 500
        
        return {
            "status": "success",
            "description": description,
            "spoken": vision.config['speech']['enabled']
        }

@api.route('/start_loop')
class StartLoop(Resource):
    @api.expect(loop_model)
    @api.response(200, 'Success', success_response)
    @api.response(500, 'Internal Server Error', error_response)
    def post(self):
        """Start continuous vision loop with periodic descriptions"""
        if not vision:
            return {"error": "Vision system not initialized"}, 500
        
        data = request.get_json() or {}
        interval = data.get('interval', vision.config['vision']['interval'])
        
        success = vision.start_continuous_vision(interval)
        if success:
            return {
                "status": "success", 
                "message": f"Started continuous vision (interval: {interval}s)"
            }
        else:
            return {"error": "Failed to start continuous vision"}, 500

@api.route('/stop_loop')
class StopLoop(Resource):
    @api.response(200, 'Success', success_response)
    @api.response(500, 'Internal Server Error', error_response)
    def post(self):
        """Stop continuous vision loop"""
        if not vision:
            return {"error": "Vision system not initialized"}, 500
        
        success = vision.stop_continuous_vision()
        if success:
            return {"status": "success", "message": "Stopped continuous vision"}
        else:
            return {"error": "Failed to stop continuous vision"}, 500

@api.route('/config')
class Config(Resource):
    @api.response(200, 'Success - Get Config')
    @api.expect(config_model)
    @api.response(200, 'Success - Update Config', success_response)
    @api.response(400, 'Bad Request', error_response)
    @api.response(500, 'Internal Server Error', error_response)
    def get(self):
        """Get current vision configuration"""
        if not vision:
            return {"error": "Vision system not initialized"}, 500
        return vision.config
    
    def post(self):
        """Update vision configuration settings"""
        if not vision:
            return {"error": "Vision system not initialized"}, 500
        
        data = request.get_json()
        if not data:
            return {"error": "No configuration data provided"}, 400
        
        # Update specific config values
        if 'interval' in data:
            vision.config['vision']['interval'] = data['interval']
        if 'first_person' in data:
            vision.config['vision']['first_person'] = data['first_person']
        if 'speech_enabled' in data:
            vision.config['speech']['enabled'] = data['speech_enabled']
        
        return {"status": "success", "message": "Configuration updated"}

@api.route('/describe_filtered')
class DescribeFiltered(Resource):
    @api.response(200, 'Success', describe_response)
    @api.response(500, 'Internal Server Error', error_response)
    def get(self):
        """Get AI description using Waldo Vision intelligent filtering"""
        if not vision:
            return {"error": "Vision system not initialized"}, 500
        
        # Use Waldo Vision filter if available
        filter_obj = streaming.filter if streaming.filter else None
        description = vision.get_filtered_view_description(filter_obj)
        
        return {
            "status": "success" if description else "no_trigger",
            "description": description or "No significant change detected - API call saved",
            "filter_used": filter_obj is not None,
            "waldo_vision_active": FILTER_AVAILABLE and streaming.filter is not None
        }

@api.route('/stream/config')
class StreamConfig(Resource):
    @api.expect(stream_config_model)
    @api.response(200, 'Success', success_response)
    @api.response(400, 'Bad Request', error_response)
    @api.response(500, 'Internal Server Error', error_response)
    def post(self):
        """Configure WebSocket streaming and filter parameters"""
        if not vision:
            return {"error": "Vision system not initialized"}, 500
        
        data = request.get_json()
        if not data:
            return {"error": "No configuration data provided"}, 400
        
        old_config = streaming.config.copy()
        
        # Update streaming configuration
        for key in ['frame_interval_ms', 'change_threshold', 'filter_enabled', 'buffer_duration_ms']:
            if key in data:
                streaming.config[key] = data[key]
        
        # Update filter if available and running
        if streaming.filter and FILTER_AVAILABLE:
            try:
                streaming.filter.configure(
                    buffer_duration_ms=streaming.config['buffer_duration_ms'],
                    change_threshold=streaming.config['change_threshold'],
                    frame_interval_ms=streaming.config['frame_interval_ms']
                )
            except Exception as e:
                logging.error(f"Filter configuration error: {e}")
        
        return {
            "status": "success", 
            "message": "Stream configuration updated",
            "old_config": old_config,
            "new_config": streaming.config,
            "filter_available": FILTER_AVAILABLE
        }

@api.route('/stream/control')
class StreamControl(Resource):
    @api.doc(params={
        'action': {'description': 'Stream control action', 'enum': ['start', 'stop'], 'required': True}
    })
    @api.response(200, 'Success', success_response)
    @api.response(400, 'Bad Request', error_response)
    @api.response(500, 'Internal Server Error', error_response)
    def post(self):
        """Start or stop WebSocket streaming with intelligent filtering"""
        if not vision:
            return {"error": "Vision system not initialized"}, 500
        
        action = request.args.get('action')
        if not action:
            return {"error": "Missing 'action' parameter"}, 400
        
        if action not in ['start', 'stop']:
            return {"error": "Action must be 'start' or 'stop'"}, 400
        
        if action == 'start':
            if streaming.streaming:
                return {"error": "Streaming already active"}, 400
            
            # Initialize filter
            filter_initialized = streaming.initialize_filter()
            
            # Start streaming would be handled by WebSocket connection
            # This endpoint just prepares the system
            return {
                "status": "success",
                "message": "Stream prepared - connect via WebSocket to start",
                "config": streaming.config,
                "filter_available": FILTER_AVAILABLE,
                "filter_initialized": filter_initialized,
                "websocket_url": "ws://raspberrypi:5002/socket.io/"
            }
        else:  # stop
            streaming.streaming = False
            return {"status": "success", "message": "Stream stopped"}

@api.route('/stream/status')
class StreamStatus(Resource):
    @api.response(200, 'Success')
    def get(self):
        """Get WebSocket streaming status and performance metrics"""
        performance = {}
        if streaming.stats['start']:
            uptime = time.time() - streaming.stats['start']
            fps = streaming.stats['frames'] / uptime if uptime > 0 else 0
            trigger_rate = (streaming.stats['triggers'] / streaming.stats['frames'] * 100) if streaming.stats['frames'] > 0 else 0
            
            performance = {
                'uptime_seconds': round(uptime, 1),
                'capture_fps': round(fps, 1),
                'frames_processed': streaming.stats['frames'],
                'ai_triggers': streaming.stats['triggers'],
                'api_calls_saved': streaming.stats['saved'],
                'trigger_rate_percent': round(trigger_rate, 1),
                'efficiency_percent': round((streaming.stats['saved'] / max(streaming.stats['frames'], 1)) * 100, 1)
            }
        
        return {
            "streaming_active": streaming.streaming,
            "client_count": streaming.client_count,
            "config": streaming.config,
            "filter_available": FILTER_AVAILABLE,
            "filter_initialized": streaming.filter is not None,
            "performance": performance,
            "camera_resolution": "4K (3840x2160) @ 30fps",
            "websocket_url": "ws://raspberrypi:5002/socket.io/"
        }

@api.route('/filter/info')
class FilterInfo(Resource):
    @api.response(200, 'Success')
    @api.response(500, 'Internal Server Error', error_response) 
    def get(self):
        """Get information about the Rust frame change detection filter"""
        filter_status = {
            "available": FILTER_AVAILABLE,
            "initialized": streaming.filter is not None,
            "purpose": "Detect significant frame changes to optimize AI API usage",
            "performance_target": "Sub-millisecond processing per frame",
            "algorithm": "Pixel comparison over sliding 100ms window"
        }
        
        if streaming.filter:
            try:
                buffer_duration, threshold, interval = streaming.filter.get_config()
                filter_status["current_config"] = {
                    "buffer_duration_ms": buffer_duration,
                    "change_threshold_percent": threshold, 
                    "frame_interval_ms": interval
                }
            except Exception as e:
                filter_status["config_error"] = str(e)
        
        return filter_status

@api.route('/monitor/start')
class MonitorStart(Resource):
    @api.response(200, 'Success', success_response)
    @api.response(500, 'Internal Server Error', error_response)
    def post(self):
        """Start continuous Waldo Vision monitoring (event-driven, no intervals!)"""
        success = waldo_monitor.start_monitoring(vision)  # Pass shared vision system
        
        if success:
            return {
                "status": "success",
                "message": "Continuous Waldo Vision monitoring started",
                "mode": "event_driven",
                "capture_rate": "30fps",
                "log_file": "/home/nerostar/Projects/corpus/waldo_vision.log",
                "monitor_command": "tail -f /home/nerostar/Projects/corpus/waldo_vision.log"
            }
        else:
            return {"error": "Failed to start monitoring"}, 500

@api.route('/monitor/stop')
class MonitorStop(Resource):
    @api.response(200, 'Success', success_response)
    @api.response(500, 'Internal Server Error', error_response)
    def post(self):
        """Stop continuous Waldo Vision monitoring"""
        success = waldo_monitor.stop_monitoring()
        
        if success:
            return {"status": "success", "message": "Continuous monitoring stopped"}
        else:
            return {"error": "Monitoring was not active"}, 500

@api.route('/monitor/status')
class MonitorStatus(Resource):
    @api.response(200, 'Success')
    def get(self):
        """Get continuous monitoring status and performance metrics"""
        status = waldo_monitor.get_status()
        
        return {
            "continuous_monitoring": status,
            "description": "Event-driven vision monitoring with Waldo Vision intelligence",
            "how_it_works": [
                "Captures frames continuously at 30fps",
                "Every frame processed through Waldo Vision filter", 
                "Auto-triggers Gemini+Speech only on significant scene changes",
                "Respects intelligent cooldowns (Volatile=1s, Disturbed=0.25s)"
            ],
            "log_monitoring": "tail -f /home/nerostar/Projects/corpus/waldo_vision.log"
        }

@api.route('/camera/config') 
class CameraConfig(Resource):
    @api.doc(params={
        'quality_preset': {'description': 'Camera quality preset', 'enum': list(QUALITY_PRESETS.keys()), 'required': False}
    })
    @api.response(200, 'Success', success_response)
    @api.response(400, 'Bad Request', error_response)
    @api.response(500, 'Internal Server Error', error_response)
    def post(self):
        """Configure camera resolution and frame rate"""
        if not vision or not vision.camera:
            return {"error": "Camera not available"}, 500
        
        preset = request.args.get('quality_preset')
        if not preset:
            return {"error": "Missing 'quality_preset' parameter"}, 400
        
        if preset not in QUALITY_PRESETS:
            return {"error": f"Invalid preset. Choose from: {list(QUALITY_PRESETS.keys())}"}, 400
        
        try:
            preset_config = QUALITY_PRESETS[preset]
            
            # Apply camera settings
            vision.camera.set(cv2.CAP_PROP_FRAME_WIDTH, preset_config['width'])
            vision.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, preset_config['height'])
            vision.camera.set(cv2.CAP_PROP_FPS, preset_config['fps'])
            
            # Verify what the camera actually applied
            actual_width = int(vision.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(vision.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = int(vision.camera.get(cv2.CAP_PROP_FPS))
            
            # Update streaming config to match actual camera settings
            actual_interval_ms = int(1000 / actual_fps) if actual_fps > 0 else preset_config['interval_ms']
            streaming.config['frame_interval_ms'] = actual_interval_ms
            streaming.config['resolution'] = {'width': actual_width, 'height': actual_height}
            
            # Update filter interval if available
            if streaming.filter:
                streaming.filter.configure(frame_interval_ms=actual_interval_ms)
            
            return {
                "status": "success",
                "message": f"Camera configured to {preset}",
                "requested_config": {
                    "resolution": f"{preset_config['width']}x{preset_config['height']}",
                    "fps": preset_config['fps'],
                    "interval_ms": preset_config['interval_ms']
                },
                "actual_config": {
                    "resolution": f"{actual_width}x{actual_height}",
                    "fps": actual_fps,
                    "interval_ms": actual_interval_ms
                },
                "fps_applied": actual_fps == preset_config['fps'],
                "resolution_applied": (actual_width == preset_config['width'] and actual_height == preset_config['height']),
                "streaming_updated": True
            }
            
        except Exception as e:
            return {"error": f"Failed to configure camera: {str(e)}"}, 500

@api.route('/camera/status')
class CameraStatus(Resource):
    @api.response(200, 'Success')
    @api.response(500, 'Internal Server Error', error_response)
    def get(self):
        """Get current camera configuration and supported resolutions"""
        if not vision or not vision.camera:
            return {"error": "Camera not available"}, 500
        
        try:
            # Get current camera settings
            current_width = int(vision.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
            current_height = int(vision.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
            current_fps = int(vision.camera.get(cv2.CAP_PROP_FPS))
            
            # Find matching preset
            current_preset = None
            for preset_name, preset_config in QUALITY_PRESETS.items():
                if (preset_config['width'] == current_width and 
                    preset_config['height'] == current_height and
                    preset_config['fps'] == current_fps):
                    current_preset = preset_name
                    break
            
            return {
                "current_resolution": f"{current_width}x{current_height}",
                "current_fps": current_fps,
                "current_preset": current_preset or "custom",
                "available_presets": {
                    preset: f"{config['width']}x{config['height']} @ {config['fps']}fps"
                    for preset, config in QUALITY_PRESETS.items()
                },
                "streaming_interval_ms": streaming.config['frame_interval_ms'],
                "camera_model": "Logitech BRIO 4K"
            }
            
        except Exception as e:
            return {"error": f"Failed to get camera status: {str(e)}"}, 500

if __name__ == '__main__':
    # Load config for server settings
    try:
        with open('config.yaml', 'r') as file:
            config = yaml.safe_load(file)
        api_config = config.get('api', {})
    except:
        api_config = {'host': '0.0.0.0', 'port': 5002}
    
    try:
        # Use SocketIO to run the app (supports both HTTP and WebSocket)
        socketio.run(
            app,
            host=api_config.get('host', '0.0.0.0'),
            port=api_config.get('port', 5002),
            debug=False,
            allow_unsafe_werkzeug=True
        )
    finally:
        if vision:
            vision.cleanup()
        if streaming.streaming:
            streaming.streaming = False