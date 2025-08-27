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

class VisionSystem:
    def __init__(self, config_path: str = "config.yaml"):
        self.config = self._load_config(config_path)
        self.camera = None
        self.continuous_thread = None
        self.continuous_running = False
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
            self.camera = cv2.VideoCapture(camera_config['device_id'])
            
            # Set camera properties
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, camera_config['resolution']['width'])
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_config['resolution']['height'])
            self.camera.set(cv2.CAP_PROP_FPS, camera_config['fps'])
            
            if self.camera.isOpened():
                logging.info("Camera initialized successfully")
            else:
                logging.error("Failed to open camera")
                self.camera = None
        except Exception as e:
            logging.error(f"Camera initialization failed: {e}")
            self.camera = None
    
    def _initialize_gemini(self):
        try:
            api_key = self.config['gemini'].get('api_key') or os.environ.get('GEMINI_API_KEY')
            if not api_key:
                logging.error("Gemini API key not found in config or environment")
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
        if not self.model:
            logging.error("Gemini model not available")
            return None
        
        try:
            # Convert OpenCV image to PIL format
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_image)
            
            # Create prompt for first-person perspective
            if self.config['vision']['first_person']:
                prompt = ("Describe what you see in this image from a first-person perspective, "
                         "as if you are an AI companion looking through your camera. "
                         "Start with 'I can see' or 'I notice' and describe the scene naturally. "
                         "Keep it concise, around 1-2 sentences.")
            else:
                prompt = "Describe what you see in this image concisely."
            
            # Generate description
            response = self.model.generate_content([prompt, pil_image])
            description = response.text.strip()
            
            logging.info(f"Generated description: {description[:50]}...")
            return description
            
        except Exception as e:
            logging.error(f"Image analysis failed: {e}")
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