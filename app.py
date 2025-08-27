from flask import Flask, request, jsonify, send_file
import logging
import yaml
import cv2
import numpy as np
import io
import base64
import time
from PIL import Image
from corpus_vision import VisionSystem

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# Initialize Vision System
try:
    vision = VisionSystem()
except Exception as e:
    logging.error(f"Failed to initialize vision system: {e}")
    vision = None

@app.route('/status', methods=['GET'])
def status():
    if not vision:
        return jsonify({"error": "Vision system not initialized"}), 500
    
    return jsonify({
        "status": "running",
        "module": "corpus-vision",
        **vision.get_status()
    })

@app.route('/capture', methods=['GET'])
def capture():
    if not vision:
        return jsonify({"error": "Vision system not initialized"}), 500
    
    image = vision.capture_image()
    if image is None:
        return jsonify({"error": "Failed to capture image"}), 500
    
    # Convert image to base64 for JSON response
    _, buffer = cv2.imencode('.jpg', image)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    
    return jsonify({
        "status": "success",
        "image": f"data:image/jpeg;base64,{img_base64}",
        "timestamp": int(time.time())
    })

@app.route('/analyze', methods=['POST'])
def analyze():
    if not vision:
        return jsonify({"error": "Vision system not initialized"}), 500
    
    # For now, analyze current camera view
    # TODO: Accept uploaded images
    image = vision.capture_image()
    if image is None:
        return jsonify({"error": "Failed to capture image"}), 500
    
    description = vision.analyze_image(image)
    if description is None:
        return jsonify({"error": "Failed to analyze image"}), 500
    
    return jsonify({
        "status": "success",
        "description": description,
        "timestamp": int(time.time())
    })

@app.route('/describe', methods=['GET'])
def describe():
    if not vision:
        return jsonify({"error": "Vision system not initialized"}), 500
    
    description = vision.get_current_view_description()
    if description is None:
        return jsonify({"error": "Failed to get description"}), 500
    
    return jsonify({
        "status": "success",
        "description": description,
        "spoken": vision.config['speech']['enabled']
    })

@app.route('/start_loop', methods=['POST'])
def start_loop():
    if not vision:
        return jsonify({"error": "Vision system not initialized"}), 500
    
    data = request.get_json() or {}
    interval = data.get('interval', vision.config['vision']['interval'])
    
    success = vision.start_continuous_vision(interval)
    if success:
        return jsonify({
            "status": "success", 
            "message": f"Started continuous vision (interval: {interval}s)"
        })
    else:
        return jsonify({"error": "Failed to start continuous vision"}), 500

@app.route('/stop_loop', methods=['POST'])
def stop_loop():
    if not vision:
        return jsonify({"error": "Vision system not initialized"}), 500
    
    success = vision.stop_continuous_vision()
    if success:
        return jsonify({"status": "success", "message": "Stopped continuous vision"})
    else:
        return jsonify({"error": "Failed to stop continuous vision"}), 500

@app.route('/config', methods=['GET', 'POST'])
def config():
    if not vision:
        return jsonify({"error": "Vision system not initialized"}), 500
    
    if request.method == 'GET':
        return jsonify(vision.config)
    
    # POST: Update configuration
    data = request.get_json()
    if not data:
        return jsonify({"error": "No configuration data provided"}), 400
    
    # Update specific config values
    if 'interval' in data:
        vision.config['vision']['interval'] = data['interval']
    if 'first_person' in data:
        vision.config['vision']['first_person'] = data['first_person']
    if 'speech_enabled' in data:
        vision.config['speech']['enabled'] = data['speech_enabled']
    
    return jsonify({"status": "success", "message": "Configuration updated"})

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