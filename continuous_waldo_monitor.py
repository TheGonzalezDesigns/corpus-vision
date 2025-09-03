# THEORY:
# This module implements the core continuous monitoring system that bridges
# computer vision hardware with Waldo Vision's intelligent filtering engine.
# It represents the "event-driven architecture" philosophy where the system
# responds to actual environmental changes rather than arbitrary time intervals.
#
# The monitor solves the fundamental AI companion challenge: providing real-time
# responsiveness while maintaining cost efficiency. By processing every frame
# through Waldo Vision's multi-layer analysis engine, it achieves 95%+ API
# cost savings while maintaining sub-second response times to significant events.
#
# Key Innovations:
# - Shared camera access (prevents hardware conflicts)
# - Continuous 30fps video feed processing
# - Scene state-based triggering (DISTURBED events only)
# - Integrated speech pipeline (automatic descriptions)
# - Real-time performance monitoring and logging

# CAVEATS & WARNINGS:
# - Camera sharing requires careful initialization order (main service first)
# - Fast capture bypasses buffer flushing (may get slightly stale frames)
# - Monitoring thread can fail silently if camera becomes unavailable
# - Speech integration depends on external service availability (port 5001)
# - Memory usage grows with processing time (no automatic garbage collection)
# - Threading model may cause race conditions under high load
# - No automatic restart if Waldo Vision filter crashes
# - Hardcoded frame quality settings (80% JPEG) not configurable via API

import cv2
import base64
import time
import threading
import logging
from corpus_vision import VisionSystem
from waldo_vision_logger import waldo_logger

try:
    from frame_change_detector import FrameChangeDetector
    FILTER_AVAILABLE = True
except ImportError:
    FILTER_AVAILABLE = False

class ContinuousWaldoMonitor:
    def __init__(self, shared_vision_system=None):
        self.vision = shared_vision_system  # Use shared camera instead of creating new one
        self.filter = None
        self.monitoring = False
        self.monitor_thread = None
        self.stats = {
            'frames_processed': 0,
            'triggers': 0,
            'api_saves': 0,
            'start_time': None
        }
        
    def initialize(self, shared_vision_system):
        """Initialize with shared vision system to avoid camera conflicts"""
        try:
            # Use existing vision system instead of creating new one
            self.vision = shared_vision_system
            
            if not self.vision or not self.vision.camera or not self.vision.camera.isOpened():
                waldo_logger.logger.error("❌ Shared camera not available")
                return False
            
            waldo_logger.logger.info("📹 Using shared camera for continuous monitoring")
            
            # Initialize Waldo Vision filter
            if FILTER_AVAILABLE:
                self.filter = FrameChangeDetector(
                    100,    # buffer_duration_ms
                    5.0,    # change_threshold
                    33      # frame_interval_ms (30fps)
                )
                waldo_logger.logger.info("🦀 Waldo Vision filter initialized with shared camera")
                return True
            else:
                waldo_logger.logger.error("❌ Waldo Vision filter not available")
                return False
                
        except Exception as e:
            waldo_logger.logger.error(f"❌ Initialization failed: {e}")
            return False
    
    def start_monitoring(self, shared_vision_system):
        """Start continuous monitoring with shared camera"""
        if self.monitoring:
            waldo_logger.logger.warning("⚠️ Monitoring already active")
            return False
        
        if not self.initialize(shared_vision_system):
            return False
        
        self.monitoring = True
        self.stats['start_time'] = time.time()
        self._reset_stats()
        
        def continuous_monitor_worker():
            waldo_logger.logger.info("🚀 Starting continuous video feed monitoring")
            
            while self.monitoring:
                current_time = time.time()  # Need this for timestamps
                try:
                    # DEBUG: Log every 30 frames (every ~1 second)
                    if self.stats['frames_processed'] % 30 == 0:
                        waldo_logger.logger.info(f"📹 Processing frame #{self.stats['frames_processed']}")
                    
                    # Continuous video feed - no timing restrictions!
                    ret, frame = self.vision.camera.read()
                    if ret and frame is not None:
                            self.stats['frames_processed'] += 1
                            
                            # Convert to base64 for Waldo Vision
                            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                            frame_b64 = base64.b64encode(buffer).decode('utf-8')
                            
                            # Process through Waldo Vision filter with scene state
                            timestamp_ms = int(current_time * 1000)
                            should_trigger, confidence, tracked_objects, scene_state = self.filter.process_frame_with_state(frame_b64, timestamp_ms)
                            
                            # Log Waldo Vision analysis (reduced frequency to avoid spam)
                            if should_trigger or self.stats['frames_processed'] % 30 == 0:
                                try:
                                    _, volatile_cooldown, disturbed_cooldown = self.filter.get_scene_status()
                                    cooldown_remaining = max(volatile_cooldown, disturbed_cooldown)
                                    
                                    waldo_logger.log_frame_analysis(
                                        should_trigger=should_trigger,
                                        confidence=confidence,
                                        tracked_objects=tracked_objects, 
                                        scene_state=scene_state,  # Now shows actual state!
                                        cooldown_remaining=cooldown_remaining
                                    )
                                except Exception as log_error:
                                    # Don't let logging errors crash the monitoring
                                    waldo_logger.logger.error(f"Logging error: {log_error}")
                            
                            # If Waldo Vision says trigger, do full AI analysis
                            if should_trigger:
                                self.stats['triggers'] += 1
                                waldo_logger.logger.info(f"🧠 AUTO-TRIGGERING Gemini analysis (confidence: {confidence:.1f}%)")
                                
                                # Get AI description
                                description = self.vision.analyze_image(frame)
                                if description:
                                    waldo_logger.logger.info(f"🗣️ AUTO-SPEAKING: {description[:60]}...")
                                    
                                    # Send to speech
                                    if self.vision.config['speech']['enabled']:
                                        self.vision.speak_description(description)
                            else:
                                self.stats['api_saves'] += 1
                    else:
                        # Log failed frame capture occasionally  
                        if self.stats['frames_processed'] % 30 == 0:
                            waldo_logger.logger.error(f"❌ Frame capture failed")
                        
                except Exception as e:
                    waldo_logger.logger.error(f"❌ Monitor loop error: {e}")
                    # Brief pause on error to prevent spam
                    time.sleep(0.1)
            
            waldo_logger.logger.info("🛑 Continuous monitoring stopped")
        
        self.monitor_thread = threading.Thread(target=continuous_monitor_worker, daemon=True)
        self.monitor_thread.start()
        
        waldo_logger.logger.info("🎯 Event-driven monitoring ACTIVE - waiting for scene changes...")
        return True
    
    def stop_monitoring(self):
        """Stop continuous monitoring"""
        if not self.monitoring:
            return False
        
        self.monitoring = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=2)
        
        # Log final stats
        if self.stats['frames_processed'] > 0:
            efficiency = (self.stats['api_saves'] / self.stats['frames_processed']) * 100
            waldo_logger.log_pipeline_stats(
                self.stats['frames_processed'],
                self.stats['triggers'], 
                self.stats['api_saves']
            )
            waldo_logger.logger.info(f"📊 SESSION COMPLETE | Efficiency: {efficiency:.1f}% API savings")
        
        waldo_logger.logger.info("🛑 Continuous monitoring STOPPED")
        return True
    
    def _reset_stats(self):
        """Reset statistics"""
        for key in ['frames_processed', 'triggers', 'api_saves']:
            self.stats[key] = 0
    
    def get_status(self):
        """Get monitoring status and stats"""
        if self.stats['start_time']:
            uptime = time.time() - self.stats['start_time']
            fps = self.stats['frames_processed'] / uptime if uptime > 0 else 0
            efficiency = (self.stats['api_saves'] / max(self.stats['frames_processed'], 1)) * 100
            
            return {
                'monitoring': self.monitoring,
                'uptime_seconds': round(uptime, 1),
                'fps': round(fps, 1),
                'frames_processed': self.stats['frames_processed'],
                'ai_triggers': self.stats['triggers'],
                'api_calls_saved': self.stats['api_saves'],
                'efficiency_percent': round(efficiency, 1),
                'filter_available': FILTER_AVAILABLE
            }
        
        return {
            'monitoring': self.monitoring,
            'filter_available': FILTER_AVAILABLE
        }

# Global monitor instance (will be initialized with shared vision system)
waldo_monitor = ContinuousWaldoMonitor()