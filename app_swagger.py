from flask import Flask, request
from flask_restx import Api, Resource, fields
import logging
import yaml
import cv2
import numpy as np
import base64
import time
from corpus_vision import VisionSystem

app = Flask(__name__)
api = Api(app,
    version='1.0', 
    title='Corpus Vision API',
    description='Computer vision and image analysis capability for Corpus AI companion',
    doc='/swagger'
)

logging.basicConfig(level=logging.INFO)

# Initialize Vision System
try:
    vision = VisionSystem()
except Exception as e:
    logging.error(f"Failed to initialize vision system: {e}")
    vision = None

# Define API models
loop_model = api.model('LoopRequest', {
    'interval': fields.Integer(description='Seconds between descriptions', example=5, default=5)
})

config_model = api.model('ConfigRequest', {
    'interval': fields.Integer(description='Seconds between descriptions', example=5),
    'first_person': fields.Boolean(description='Use first-person perspective', example=True),
    'speech_enabled': fields.Boolean(description='Enable speech output', example=True)
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

if __name__ == '__main__':
    # Load config for server settings
    try:
        with open('config.yaml', 'r') as file:
            config = yaml.safe_load(file)
        api_config = config.get('api', {})
    except:
        api_config = {'host': '0.0.0.0', 'port': 5002}
    
    try:
        app.run(
            host=api_config.get('host', '0.0.0.0'),
            port=api_config.get('port', 5002),
            debug=False
        )
    finally:
        if vision:
            vision.cleanup()