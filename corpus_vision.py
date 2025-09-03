# THEORY:
# This module represents the core "eyes" capability of the Corpus AI companion.
# It bridges physical camera hardware with advanced AI vision analysis through
# Google's Gemini AI, implementing both single-shot analysis and continuous
# monitoring workflows.
#
# The module embodies the "sensor-to-intelligence" pipeline, transforming raw
# camera pixels into meaningful, first-person descriptions that the AI companion
# can speak about its environment. It handles the critical camera buffer
# management needed for live image capture while providing both REST API
# endpoints and integration points for intelligent filtering systems.
#
# Key Responsibilities:
# - Camera hardware abstraction and configuration
# - Image capture with buffer flushing for live feeds
# - Gemini AI integration for scene understanding
# - Speech API integration for autonomous descriptions
# - Support for both filtered (Waldo Vision) and direct analysis modes
#
# This module is the foundation that gives the Corpus AI companion the
# ability to "see" and understand its environment with human-like perception.

# CAVEATS & WARNINGS:
# - Camera device IDs are hardcoded in config (may change after USB reconnection)
# - Buffer flushing (5 frames) adds significant latency to single captures
# - Gemini API key required in environment (GEMINI_API_KEY) - fails without it
# - No camera reconnection logic if device becomes unavailable during operation
# - Speech integration assumes localhost:5001 (hardcoded, not configurable)
# - Continuous vision mode not integrated with Waldo Vision (legacy implementation)
# - No image quality validation before sending to Gemini (may waste API calls)
# - Camera configuration changes require service restart
# - First-person description prompts not customizable via API

import cv2
import yaml
import logging
import base64
import time
import threading
import os
from typing import Optional, Dict, Any, Tuple
from PIL import Image
import numpy as np
import google.generativeai as genai
import requests
import io
from provider_router import VisionRouter

class VisionSystem:
    def __init__(self, config_path: str = "config.yaml"):
        self.config = self._load_config(config_path)
        self.camera = None
        self.continuous_thread = None
        self.continuous_running = False
        # Ensure model attribute always exists to avoid AttributeError
        self.model = None
        self._initialize_camera()
        self._initialize_gemini()
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            logging.warning(f"Config file {config_path} not found, using defaults")
            return self._default_config()
    
    def _default_config(self) -> Dict[str, Any]:
        return {
            'camera': {'device_id': 0, 'resolution': {'width': 1920, 'height': 1080}, 'fps': 30},
            'gemini': {'model': 'gemini-1.5-flash', 'max_tokens': 150, 'temperature': 0.7},
            'vision': {'continuous_mode': False, 'interval': 5, 'first_person': True},
            'speech': {'enabled': True, 'speech_api_url': 'http://localhost:5001'}
        }
    
    def _initialize_camera(self):
        try:
            camera_config = self.config['camera']
            # Try configured device first
            self.camera = cv2.VideoCapture(camera_config.get('device_id', 0))
            
            # Set camera properties
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, camera_config['resolution']['width'])
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_config['resolution']['height'])
            self.camera.set(cv2.CAP_PROP_FPS, camera_config['fps'])
            
            if self.camera.isOpened():
                logging.info("Camera initialized successfully")
            else:
                logging.warning("Configured camera not available. Attempting auto-detection...")
                if self.camera:
                    try:
                        self.camera.release()
                    except Exception:
                        pass
                self.camera = None
                # Auto-detect camera device 0-5
                for dev_id in range(0, 6):
                    cam = cv2.VideoCapture(dev_id)
                    if cam.isOpened():
                        # Apply same settings
                        cam.set(cv2.CAP_PROP_FRAME_WIDTH, camera_config['resolution']['width'])
                        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_config['resolution']['height'])
                        cam.set(cv2.CAP_PROP_FPS, camera_config['fps'])
                        self.camera = cam
                        self.config['camera']['device_id'] = dev_id
                        logging.info(f"Auto-selected camera device_id={dev_id}")
                        break
                if not self.camera:
                    logging.error("Failed to open any camera device (0-5)")
        except Exception as e:
            logging.error(f"Camera initialization failed: {e}")
            self.camera = None
    
    def _initialize_gemini(self):
        try:
            api_key = self.config['gemini'].get('api_key') or os.environ.get('GEMINI_API_KEY')
            if not api_key:
                logging.error("Gemini API key not found in config or environment")
                self.model = None
                return
                
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(self.config['gemini']['model'])
            logging.info("Gemini AI initialized successfully")
        except Exception as e:
            logging.error(f"Gemini initialization failed: {e}")
            self.model = None
    
    def capture_image(self) -> Optional[np.ndarray]:
        if not self.camera or not self.camera.isOpened():
            logging.error("Camera not available")
            return None
        
        # Flush camera buffer by reading multiple frames to get fresh image
        # This ensures we get the current live view, not a buffered frame
        for _ in range(5):
            ret, frame = self.camera.read()
            if not ret:
                logging.error("Failed to capture frame during buffer flush")
                return None
        
        # Final read for the actual image we want
        ret, frame = self.camera.read()
        if ret:
            logging.info("Fresh image captured successfully")
            return frame
        else:
            logging.error("Failed to capture final image")
            return None
    
    def analyze_image(self, image: np.ndarray) -> Optional[str]:
        # Convert OpenCV image to PIL
        try:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_image)
        except Exception as e:
            logging.error(f"Image conversion failed: {e}")
            return None

        if self.config['vision']['first_person']:
            prompt = ("Describe what you see in this image from a first-person perspective, "
                      "as if you are an AI companion looking through your camera. "
                      "Start with 'I can see' or 'I notice' and describe the scene naturally. "
                      "Keep it concise, around 1-2 sentences.")
        else:
            prompt = "Describe what you see in this image concisely."

        # Use router with priority order (env VISION_PROVIDER_ORDER)
        try:
            router = getattr(self, 'vision_router', None)
            if router is None:
                router = VisionRouter()
                self.vision_router = router
            desc = router.analyze(pil_image, prompt)
            if desc:
                return desc
        except Exception as e:
            logging.warning(f"Vision router failed: {e}")

        # Fallback to Gemini direct if available
        try:
            if self.model is not None:
                resp = self.model.generate_content([prompt, pil_image])
                return (getattr(resp, 'text', '') or '').strip()
        except Exception as e:
            logging.error(f"Gemini fallback failed: {e}")
        return None

    def speak_description(self, description: str) -> bool:
        if not self.config['speech']['enabled']:
            return False
        
        try:
            speech_url = f"{self.config['speech']['speech_api_url']}/speak"
            response = requests.post(
                speech_url,
                json={'text': description},
                timeout=10
            )
            return response.status_code == 200
        except Exception as e:
            logging.error(f"Failed to speak description: {e}")
            return False
    
    def get_current_view_description(self) -> Optional[str]:
        image = self.capture_image()
        if image is None:
            return None
        
        description = self.analyze_image(image)
        if description and self.config['speech']['enabled']:
            self.speak_description(description)
        
        return description
    
    def get_filtered_view_description(self, filter_obj=None) -> Optional[str]:
        """Get description using Waldo Vision filter to determine if analysis is needed"""
        from waldo_vision_logger import waldo_logger
        
        # Capture frame
        image = self.capture_image()
        if image is None:
            waldo_logger.logger.info("❌ Frame capture failed")
            return None
        
        # Use Waldo Vision filter if available
        if filter_obj:
            try:
                # Convert frame to base64 for filter
                _, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 80])
                frame_b64 = base64.b64encode(buffer).decode('utf-8')
                
                # Process through Waldo Vision
                timestamp_ms = int(time.time() * 1000)
                should_trigger, confidence, tracked_objects = filter_obj.process_frame(frame_b64, timestamp_ms)
                
                # Log Waldo Vision decision
                waldo_logger.log_frame_analysis(
                    should_trigger=should_trigger,
                    confidence=confidence, 
                    tracked_objects=tracked_objects,
                    scene_state="Active",  # Would need to get actual state from filter
                    cooldown_remaining=0.0  # Would need to get from filter
                )
                
                # Only proceed with AI analysis if Waldo Vision says so
                if should_trigger:
                    waldo_logger.logger.info(f"🧠 Triggering Gemini analysis (confidence: {confidence:.1f}%)")
                    description = self.analyze_image(image)
                    
                    if description:
                        waldo_logger.logger.info(f"🗣️ Speaking: {description[:50]}...")
                        if self.config['speech']['enabled']:
                            self.speak_description(description)
                        return description
                else:
                    waldo_logger.logger.info(f"⏸️ No trigger - saving API call (confidence: {confidence:.1f}%)")
                    return None
                    
            except Exception as e:
                waldo_logger.logger.error(f"❌ Filter error: {e}")
                # Fall back to normal analysis
                pass
        
        # Fallback to normal analysis if no filter
        description = self.analyze_image(image)
        if description and self.config['speech']['enabled']:
            self.speak_description(description)
        
        return description
    
    def _continuous_vision_loop(self):
        while self.continuous_running:
            try:
                description = self.get_current_view_description()
                if description:
                    logging.info(f"Vision: {description}")
                time.sleep(self.config['vision']['interval'])
            except Exception as e:
                logging.error(f"Error in continuous vision loop: {e}")
                time.sleep(1)
    
    def start_continuous_vision(self, interval: Optional[int] = None) -> bool:
        if self.continuous_running:
            logging.warning("Continuous vision already running")
            return False
        
        if interval:
            self.config['vision']['interval'] = interval
        
        self.continuous_running = True
        self.continuous_thread = threading.Thread(target=self._continuous_vision_loop)
        self.continuous_thread.daemon = True
        self.continuous_thread.start()
        
        logging.info(f"Started continuous vision (interval: {self.config['vision']['interval']}s)")
        return True
    
    def stop_continuous_vision(self) -> bool:
        if not self.continuous_running:
            logging.warning("Continuous vision not running")
            return False
        
        self.continuous_running = False
        if self.continuous_thread:
            self.continuous_thread.join(timeout=5)
        
        logging.info("Stopped continuous vision")
        return True
    
    def get_status(self) -> Dict[str, Any]:
        return {
            'camera_available': self.camera is not None and self.camera.isOpened(),
            'gemini_available': self.model is not None,
            'continuous_running': self.continuous_running,
            'speech_enabled': self.config['speech']['enabled']
        }
    
    def cleanup(self):
        self.stop_continuous_vision()
        if self.camera:
            self.camera.release()
        cv2.destroyAllWindows()
