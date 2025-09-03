# THEORY:
# This module provides specialized logging infrastructure for the Waldo Vision
# intelligent filtering system. It creates human-readable, real-time monitoring
# of the AI companion's decision-making process, making the "black box" of
# computer vision transparent and debuggable.
#
# The logger serves multiple critical functions:
# - Performance monitoring (frames processed, API calls saved)
# - Decision transparency (why did it trigger or not trigger?)
# - Scene state tracking (Stable → Volatile → Disturbed transitions)
# - Cost optimization verification (efficiency metrics)
# - Development debugging (cooldown timers, confidence scores)
#
# By providing detailed logs of every Waldo Vision decision, users can
# understand and tune the AI companion's behavior, building trust through
# transparency in the intelligent filtering process.

# CAVEATS & WARNINGS:
# - Log file path is hardcoded (/home/nerostar/Projects/corpus/waldo_vision.log)
# - No log rotation (file grows indefinitely with continuous monitoring)
# - Timestamp formatting uses deprecated %f that doesn't work (shows %f literally)
# - No log level configuration (always INFO level)
# - Multiple logger instances may cause duplicate entries
# - No structured logging (JSON) for programmatic analysis
# - Performance impact not measured (logging in tight monitoring loop)
# - No log compression or archival for long-term storage

import logging
from logging.handlers import TimedRotatingFileHandler
import json
import time
from datetime import datetime

class WaldoVisionLogger:
    def __init__(self, log_file="/home/nerostar/Projects/corpus/waldo_vision.log"):
        self.log_file = log_file
        
        # Create dedicated logger for Waldo Vision
        self.logger = logging.getLogger("WaldoVision")
        self.logger.setLevel(logging.INFO)
        
        # Remove existing handlers to avoid duplicates
        self.logger.handlers.clear()
        
        # Create rotating file handler with daily rotation and 7 backups
        file_handler = TimedRotatingFileHandler(log_file, when='midnight', interval=1, backupCount=7, encoding='utf-8')
        file_handler.setLevel(logging.INFO)

        # Correct timestamp formatting with milliseconds (logging doesn't support %f)
        formatter = logging.Formatter('%(asctime)s.%(msecs)03d | %(message)s', datefmt='%H:%M:%S')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        # Log startup
        self.logger.info("🔥 Waldo Vision monitoring started")
        self.logger.info("=" * 60)
    
    def log_frame_analysis(self, should_trigger, confidence, tracked_objects, scene_state="Unknown", cooldown_remaining=0.0):
        """Log Waldo Vision frame analysis results"""
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        
        # Determine action emoji
        if should_trigger:
            action = "🚨 TRIGGER" if confidence > 80 else "⚠️  TRIGGER"
        else:
            action = "✅ SILENT " if cooldown_remaining == 0 else "⏳ COOLDOWN"
        
        # Create status line
        status_line = f"{action} | State: {scene_state:>10} | Confidence: {confidence:5.1f}% | Objects: {tracked_objects:2d}"
        
        if cooldown_remaining > 0:
            status_line += f" | Cooldown: {cooldown_remaining:.2f}s"
        
        self.logger.info(status_line)
        
        # Log significant events with more detail
        if should_trigger and confidence > 90:
            self.logger.info(f"🚨 HIGH PRIORITY EVENT | Confidence: {confidence:.1f}% | Scene: {scene_state}")
    
    def log_scene_transition(self, old_state, new_state):
        """Log scene state changes"""
        self.logger.info(f"🔄 STATE CHANGE: {old_state} → {new_state}")
    
    def log_api_call(self, api_type, duration_ms, success=True):
        """Log API calls and performance"""
        status = "✅" if success else "❌"
        self.logger.info(f"{status} {api_type} API | Duration: {duration_ms}ms")
    
    def log_cooldown_skip(self, scene_state, time_remaining):
        """Log when triggers are skipped due to cooldown"""
        self.logger.info(f"⏳ COOLDOWN SKIP | {scene_state} | Remaining: {time_remaining:.2f}s")
    
    def log_pipeline_stats(self, frames_processed, triggers, api_saves):
        """Log periodic pipeline statistics"""
        efficiency = (api_saves / max(frames_processed, 1)) * 100
        self.logger.info(f"📊 STATS | Frames: {frames_processed} | Triggers: {triggers} | Efficiency: {efficiency:.1f}%")

# Global logger instance
waldo_logger = WaldoVisionLogger()
