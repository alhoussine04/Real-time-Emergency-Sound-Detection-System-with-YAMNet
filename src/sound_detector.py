"""
Sound detection and filtering system for target audio events.
"""

import numpy as np
import logging
import time
from typing import List, Dict, Tuple, Optional, Callable
from datetime import datetime
from dataclasses import dataclass

from .yamnet_classifier import YAMNetClassifier
from .telegram_notifier import TelegramNotifier


@dataclass
class DetectionEvent:
    """Data class for sound detection events."""
    sound_type: str
    confidence: float
    timestamp: datetime
    top_predictions: List[Tuple[str, float]]


class SoundDetector:
    """
    Main sound detection system that processes audio and triggers notifications.
    """
    
    def __init__(
        self,
        yamnet_classifier: YAMNetClassifier,
        telegram_notifier: TelegramNotifier,
        target_sounds: List[str],
        confidence_threshold: float = 0.3,
        detection_cooldown: float = 5.0
    ):
        """
        Initialize the sound detector.
        
        Args:
            yamnet_classifier: YAMNet classifier instance
            telegram_notifier: Telegram notifier instance
            target_sounds: List of target sound class names
            confidence_threshold: Minimum confidence for detection
            detection_cooldown: Minimum time between detections (seconds)
        """
        self.classifier = yamnet_classifier
        self.notifier = telegram_notifier
        self.target_sounds = target_sounds
        self.confidence_threshold = confidence_threshold
        self.detection_cooldown = detection_cooldown
        
        # Detection state
        self.last_detection_times: Dict[str, float] = {}
        self.detection_history: List[DetectionEvent] = []
        self.total_processed = 0
        self.total_detections = 0
        
        # Callbacks
        self.detection_callback: Optional[Callable[[DetectionEvent], None]] = None
        
        self.logger = logging.getLogger(__name__)
        
        # Validate target sounds
        self._validate_target_sounds()
    
    def _validate_target_sounds(self) -> None:
        """Validate that target sounds exist in YAMNet class labels."""
        available_classes = self.classifier.get_available_classes()
        
        valid_sounds = []
        for target in self.target_sounds:
            # Check for exact match or partial match
            exact_match = target in available_classes
            partial_matches = [cls for cls in available_classes if target.lower() in cls.lower()]
            
            if exact_match:
                valid_sounds.append(target)
                self.logger.info(f"Target sound '{target}' found (exact match)")
            elif partial_matches:
                # Use the first partial match
                matched_class = partial_matches[0]
                valid_sounds.append(matched_class)
                self.logger.info(f"Target sound '{target}' mapped to '{matched_class}'")
            else:
                self.logger.warning(f"Target sound '{target}' not found in YAMNet classes")
        
        self.target_sounds = valid_sounds
        self.logger.info(f"Monitoring {len(self.target_sounds)} target sound types")
    
    def process_audio(self, audio_data: np.ndarray) -> Optional[DetectionEvent]:
        """
        Process audio data and check for target sounds.
        
        Args:
            audio_data: Audio waveform as numpy array
            
        Returns:
            DetectionEvent if target sound detected, None otherwise
        """
        self.total_processed += 1
        
        try:
            # Get predictions from YAMNet
            class_names, confidence_scores = self.classifier.classify(audio_data)
            
            if not class_names:
                self.logger.warning("No predictions returned from classifier")
                return None
            
            # Check for target sounds
            detection_event = self._check_target_sounds(class_names, confidence_scores)
            
            if detection_event:
                self.total_detections += 1
                self.detection_history.append(detection_event)
                
                # Limit history size
                if len(self.detection_history) > 1000:
                    self.detection_history = self.detection_history[-500:]
                
                # Call detection callback if set
                if self.detection_callback:
                    self.detection_callback(detection_event)
                
                # Send notification
                self._send_notification(detection_event)
                
                self.logger.info(
                    f"Detection: {detection_event.sound_type} "
                    f"(confidence: {detection_event.confidence:.3f})"
                )
            
            return detection_event
            
        except Exception as e:
            self.logger.error(f"Error processing audio: {e}")
            return None
    
    def _check_target_sounds(
        self,
        class_names: List[str],
        confidence_scores: List[float]
    ) -> Optional[DetectionEvent]:
        """
        Check if any target sounds are detected with sufficient confidence.
        
        Args:
            class_names: List of predicted class names
            confidence_scores: List of confidence scores
            
        Returns:
            DetectionEvent if target sound detected, None otherwise
        """
        current_time = time.time()
        
        # Create predictions list
        top_predictions = list(zip(class_names, confidence_scores))
        
        # Check each prediction
        for class_name, confidence in top_predictions:
            # Check if this is a target sound
            for target_sound in self.target_sounds:
                if self._is_target_match(class_name, target_sound):
                    # Check confidence threshold
                    if confidence >= self.confidence_threshold:
                        # Check cooldown
                        last_time = self.last_detection_times.get(target_sound, 0)
                        if current_time - last_time >= self.detection_cooldown:
                            # Valid detection
                            self.last_detection_times[target_sound] = current_time
                            
                            return DetectionEvent(
                                sound_type=target_sound,
                                confidence=confidence,
                                timestamp=datetime.now(),
                                top_predictions=top_predictions[:5]  # Top 5 predictions
                            )
                        else:
                            self.logger.debug(
                                f"Skipping {target_sound} detection due to cooldown "
                                f"({current_time - last_time:.1f}s < {self.detection_cooldown}s)"
                            )
        
        return None
    
    def _is_target_match(self, class_name: str, target_sound: str) -> bool:
        """
        Check if a class name matches a target sound.
        
        Args:
            class_name: YAMNet class name
            target_sound: Target sound name
            
        Returns:
            True if they match, False otherwise
        """
        # Exact match
        if class_name == target_sound:
            return True
        
        # Case-insensitive partial match
        class_lower = class_name.lower()
        target_lower = target_sound.lower()
        
        # Check if target is contained in class name or vice versa
        return target_lower in class_lower or class_lower in target_lower
    
    def _send_notification(self, detection_event: DetectionEvent) -> None:
        """
        Send notification for detection event.
        
        Args:
            detection_event: Detection event to notify about
        """
        try:
            success = self.notifier.send_notification(
                sound_type=detection_event.sound_type,
                confidence=detection_event.confidence,
                timestamp=detection_event.timestamp
            )
            
            if not success:
                self.logger.warning(f"Failed to send notification for {detection_event.sound_type}")
                
        except Exception as e:
            self.logger.error(f"Error sending notification: {e}")
    
    def set_detection_callback(self, callback: Callable[[DetectionEvent], None]) -> None:
        """
        Set callback function for detection events.
        
        Args:
            callback: Function to call when sound is detected
        """
        self.detection_callback = callback
    
    def get_detection_stats(self) -> Dict:
        """
        Get detection statistics.
        
        Returns:
            Dictionary with detection statistics
        """
        current_time = time.time()
        
        # Count detections by type
        detection_counts = {}
        for event in self.detection_history:
            sound_type = event.sound_type
            detection_counts[sound_type] = detection_counts.get(sound_type, 0) + 1
        
        # Recent detections (last hour)
        recent_detections = [
            event for event in self.detection_history
            if (datetime.now() - event.timestamp).total_seconds() < 3600
        ]
        
        stats = {
            'total_processed': self.total_processed,
            'total_detections': self.total_detections,
            'detection_rate': self.total_detections / max(self.total_processed, 1),
            'target_sounds': self.target_sounds,
            'confidence_threshold': self.confidence_threshold,
            'detection_cooldown': self.detection_cooldown,
            'detection_counts': detection_counts,
            'recent_detections_count': len(recent_detections),
            'last_detection_times': {
                sound: datetime.fromtimestamp(timestamp).isoformat()
                for sound, timestamp in self.last_detection_times.items()
            }
        }
        
        return stats
    
    def get_recent_detections(self, limit: int = 10) -> List[DetectionEvent]:
        """
        Get recent detection events.
        
        Args:
            limit: Maximum number of events to return
            
        Returns:
            List of recent detection events
        """
        return self.detection_history[-limit:] if self.detection_history else []
    
    def clear_detection_history(self) -> None:
        """Clear detection history."""
        self.detection_history.clear()
        self.last_detection_times.clear()
        self.total_detections = 0
        self.logger.info("Detection history cleared")
    
    def update_target_sounds(self, new_target_sounds: List[str]) -> None:
        """
        Update the list of target sounds.
        
        Args:
            new_target_sounds: New list of target sound names
        """
        self.target_sounds = new_target_sounds
        self._validate_target_sounds()
        self.logger.info(f"Updated target sounds: {self.target_sounds}")
    
    def update_confidence_threshold(self, new_threshold: float) -> None:
        """
        Update the confidence threshold.
        
        Args:
            new_threshold: New confidence threshold (0.0 to 1.0)
        """
        if 0.0 <= new_threshold <= 1.0:
            self.confidence_threshold = new_threshold
            self.logger.info(f"Updated confidence threshold to {new_threshold}")
        else:
            self.logger.warning(f"Invalid confidence threshold: {new_threshold}")
