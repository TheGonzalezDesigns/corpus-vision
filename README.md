# Corpus Vision - Computer Vision Module

Computer vision and image analysis capability for the Corpus AI companion system.

## Overview

This module provides real-time visual perception using a USB camera and Gemini AI for image analysis. It captures images, analyzes them with AI, and describes what it sees in first-person perspective.

## Features

- USB camera integration (Logitech 4K support)
- Real-time image capture
- Gemini AI vision analysis
- First-person perspective descriptions
- Continuous vision loop
- Integration with Corpus speech system
- REST API for vision services

## Hardware Requirements

- Raspberry Pi with USB port
- USB camera (tested with Logitech 4K)
- Internet connection for Gemini API

## Installation

```bash
# Install system dependencies
sudo apt update
sudo apt install python3-opencv

# Install Python dependencies
pip install -r requirements.txt
```

## Usage

### Standalone Vision Analysis
```python
from corpus_vision import VisionSystem

vision = VisionSystem()
image = vision.capture_image()
description = vision.analyze_image(image)
print(description)
```

### Continuous Vision Loop
```python
vision = VisionSystem()
vision.start_continuous_vision(interval=5)  # Describe every 5 seconds
```

### API Server
```bash
python app.py
```

## API

- `GET /capture` - Capture and return current image
- `POST /analyze` - Analyze uploaded image
- `GET /describe` - Get current view description
- `POST /start_loop` - Start continuous vision loop
- `POST /stop_loop` - Stop continuous vision loop
- `GET /status` - Get module status

## Configuration

See `config.yaml` for camera settings, Gemini API configuration, and speech integration parameters.

## Integration

Communicates with Corpus speech module to speak descriptions and main orchestrator for coordination.