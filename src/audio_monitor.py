"""
Main audio monitoring application that integrates all components.
"""

import os
import sys
import signal
import logging
import time
import threading
from typing import Optional, List
from datetime import datetime
from dotenv import load_dotenv
import colorlog

from .yamnet_classifier import YAMNetClassifier
from .audio_capture import AudioCapture
from .telegram_notifier import TelegramNotifier
from .sound_detector import SoundDetector, DetectionEvent


class AudioMonitor:
    """
    Main audio monitoring application.
    """
    
    def __init__(self, config_file: str = "config.env"):
        """
        Initialize the audio monitor.
        
        Args:
            config_file: Path to configuration file
        """
        self.config_file = config_file
        self.is_running = False
        self.shutdown_event = threading.Event()
        
        # Components
        self.classifier: Optional[YAMNetClassifier] = None
        self.audio_capture: Optional[AudioCapture] = None
        self.notifier: Optional[TelegramNotifier] = None
        self.detector: Optional[SoundDetector] = None
        
        # Configuration
        self.config = {}
        
        # Statistics
        self.start_time: Optional[datetime] = None
        self.stats_lock = threading.Lock()
        
        self.logger = self._setup_logging()
        
        # Load configuration
        self._load_config()
        
        # Setup signal handlers
        self._setup_signal_handlers()
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        # Create colored formatter (without emojis for Windows compatibility)
        formatter = colorlog.ColoredFormatter(
            '%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'red,bg_white',
            }
        )

        # Setup console handler with UTF-8 encoding
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)

        # Try to set UTF-8 encoding for Windows
        try:
            import sys
            if sys.platform.startswith('win'):
                import codecs
                sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
                sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')
        except:
            pass  # Fallback to default encoding

        # Setup root logger
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        logger.addHandler(console_handler)

        return logging.getLogger(__name__)
    
    def _load_config(self) -> None:
        """Load configuration from environment file."""
        try:
            # Load environment variables
            load_dotenv(self.config_file)
            
            self.config = {
                # Telegram settings
                'telegram_bot_token': os.getenv('TELEGRAM_BOT_TOKEN', ''),
                'telegram_chat_id': os.getenv('TELEGRAM_CHAT_ID', ''),
                
                # Audio settings
                'sample_rate': int(os.getenv('SAMPLE_RATE', '16000')),
                'chunk_size': int(os.getenv('CHUNK_SIZE', '1024')),
                'channels': int(os.getenv('CHANNELS', '1')),
                'audio_device_index': int(os.getenv('AUDIO_DEVICE_INDEX', '-1')),
                
                # Detection settings
                'confidence_threshold': float(os.getenv('CONFIDENCE_THRESHOLD', '0.3')),
                'detection_cooldown': float(os.getenv('DETECTION_COOLDOWN', '5.0')),
                'buffer_duration': float(os.getenv('BUFFER_DURATION', '0.975')),
                
                # Target sounds
                'target_sounds': [
                    sound.strip() 
                    for sound in os.getenv('TARGET_SOUNDS', '').split(';')
                    if sound.strip()
                ],
                
                # Logging
                'log_level': os.getenv('LOG_LEVEL', 'INFO'),
                'log_file': os.getenv('LOG_FILE', 'audio_monitor.log'),
                
                # Model path
                'model_path': os.getenv('MODEL_PATH', '1.tflite')
            }
            
            # Validate required settings
            if not self.config['telegram_bot_token']:
                raise ValueError("TELEGRAM_BOT_TOKEN is required")
            if not self.config['telegram_chat_id']:
                raise ValueError("TELEGRAM_CHAT_ID is required")
            if not self.config['target_sounds']:
                raise ValueError("TARGET_SOUNDS is required")
            
            # Update log level
            log_level = getattr(logging, self.config['log_level'].upper(), logging.INFO)
            logging.getLogger().setLevel(log_level)
            
            # Add file handler if specified
            if self.config['log_file']:
                file_handler = logging.FileHandler(self.config['log_file'])
                file_handler.setFormatter(logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                ))
                logging.getLogger().addHandler(file_handler)
            
            self.logger.info("Configuration loaded successfully")
            self.logger.info(f"Target sounds: {self.config['target_sounds']}")
            
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {e}")
            raise
    
    def _setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            self.logger.info(f"Received signal {signum}, shutting down...")
            self.stop()
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def _initialize_components(self) -> None:
        """Initialize all system components."""
        try:
            # Initialize YAMNet classifier
            self.logger.info("Initializing YAMNet classifier...")
            self.classifier = YAMNetClassifier(self.config['model_path'])
            
            # Initialize Telegram notifier
            self.logger.info("Initializing Telegram notifier...")
            self.notifier = TelegramNotifier(
                bot_token=self.config['telegram_bot_token'],
                chat_id=self.config['telegram_chat_id'],
                cooldown_seconds=self.config['detection_cooldown']
            )
            
            # Test Telegram connection
            if not self.notifier.test_connection():
                raise RuntimeError("Failed to connect to Telegram")
            
            # Initialize audio capture
            self.logger.info("Initializing audio capture...")
            self.audio_capture = AudioCapture(
                sample_rate=self.config['sample_rate'],
                chunk_size=self.config['chunk_size'],
                channels=self.config['channels'],
                device_index=self.config['audio_device_index'],
                buffer_duration=self.config['buffer_duration']
            )
            
            # Initialize sound detector
            self.logger.info("Initializing sound detector...")
            self.detector = SoundDetector(
                yamnet_classifier=self.classifier,
                telegram_notifier=self.notifier,
                target_sounds=self.config['target_sounds'],
                confidence_threshold=self.config['confidence_threshold'],
                detection_cooldown=self.config['detection_cooldown']
            )
            
            # Set detection callback
            self.detector.set_detection_callback(self._on_detection)
            
            self.logger.info("All components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize components: {e}")
            raise
    
    def _on_detection(self, detection_event: DetectionEvent) -> None:
        """
        Callback for detection events.
        
        Args:
            detection_event: Detection event data
        """
        self.logger.info(
            f"*** DETECTION: {detection_event.sound_type} "
            f"(confidence: {detection_event.confidence:.3f}) ***"
        )
        
        # Log top predictions
        for i, (class_name, score) in enumerate(detection_event.top_predictions[:3]):
            self.logger.debug(f"  {i+1}. {class_name}: {score:.3f}")
    
    def start(self) -> None:
        """Start the audio monitoring system."""
        if self.is_running:
            self.logger.warning("Audio monitor is already running")
            return
        
        try:
            self.logger.info("Starting audio monitoring system...")
            
            # Initialize components
            self._initialize_components()
            
            # Send startup notification
            self.notifier.send_startup_message()
            
            # Start audio capture
            self.audio_capture.start_capture(self._process_audio)
            
            # Set running state
            self.is_running = True
            self.start_time = datetime.now()
            
            self.logger.info("*** Audio monitoring system started successfully! ***")
            self.logger.info("Press Ctrl+C to stop...")
            
            # Main monitoring loop
            self._monitoring_loop()
            
        except Exception as e:
            self.logger.error(f"Failed to start audio monitoring: {e}")
            self.stop()
            raise
    
    def _process_audio(self, audio_data) -> None:
        """
        Process audio data through the detection pipeline.
        
        Args:
            audio_data: Audio data from capture system
        """
        if not self.is_running or self.shutdown_event.is_set():
            return
        
        try:
            # Process through detector
            self.detector.process_audio(audio_data)
            
        except Exception as e:
            self.logger.error(f"Error processing audio: {e}")
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        stats_interval = 60  # Print stats every 60 seconds
        last_stats_time = time.time()
        
        try:
            while self.is_running and not self.shutdown_event.is_set():
                # Check if it's time to print stats
                current_time = time.time()
                if current_time - last_stats_time >= stats_interval:
                    self._print_stats()
                    last_stats_time = current_time
                
                # Sleep briefly
                time.sleep(1.0)
                
        except KeyboardInterrupt:
            self.logger.info("Received keyboard interrupt")
        except Exception as e:
            self.logger.error(f"Error in monitoring loop: {e}")
        finally:
            self.stop()
    
    def _print_stats(self) -> None:
        """Print system statistics."""
        try:
            if not self.detector:
                return
            
            stats = self.detector.get_detection_stats()
            uptime = (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
            
            self.logger.info(
                f"STATS - Uptime: {uptime/3600:.1f}h, "
                f"Processed: {stats['total_processed']}, "
                f"Detections: {stats['total_detections']}, "
                f"Rate: {stats['detection_rate']:.4f}"
            )
            
            # Log audio level
            if self.audio_capture:
                audio_level = self.audio_capture.get_audio_level()
                self.logger.debug(f"Audio level: {audio_level:.4f}")
            
        except Exception as e:
            self.logger.error(f"Error printing stats: {e}")
    
    def stop(self) -> None:
        """Stop the audio monitoring system."""
        if not self.is_running:
            return
        
        self.logger.info("Stopping audio monitoring system...")
        
        # Set shutdown flag
        self.is_running = False
        self.shutdown_event.set()
        
        # Stop audio capture
        if self.audio_capture:
            self.audio_capture.stop_capture()
        
        # Send shutdown notification
        if self.notifier:
            self.notifier.send_shutdown_message()
        
        # Cleanup components
        if self.audio_capture:
            self.audio_capture.cleanup()
        
        self.logger.info("Audio monitoring system stopped")
    
    def get_status(self) -> dict:
        """Get current system status."""
        status = {
            'is_running': self.is_running,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'uptime_seconds': (datetime.now() - self.start_time).total_seconds() if self.start_time else 0,
            'config': self.config.copy()
        }
        
        # Remove sensitive information
        status['config'].pop('telegram_bot_token', None)
        
        if self.detector:
            status['detection_stats'] = self.detector.get_detection_stats()
        
        if self.audio_capture:
            status['audio_active'] = self.audio_capture.is_active()
            status['audio_level'] = self.audio_capture.get_audio_level()
        
        return status


def main():
    """Main entry point for the audio monitoring system."""
    import argparse

    parser = argparse.ArgumentParser(description='Real-time Audio Monitoring with YAMNet')
    parser.add_argument(
        '--config',
        default='config.env',
        help='Path to configuration file (default: config.env)'
    )
    parser.add_argument(
        '--model',
        default='1.tflite',
        help='Path to YAMNet TF-Lite model file (default: 1.tflite)'
    )
    parser.add_argument(
        '--test-telegram',
        action='store_true',
        help='Test Telegram connection and exit'
    )
    parser.add_argument(
        '--list-devices',
        action='store_true',
        help='List available audio devices and exit'
    )

    args = parser.parse_args()

    # Handle special commands
    if args.list_devices:
        from .audio_capture import AudioCapture
        try:
            capture = AudioCapture()
            capture.cleanup()
        except Exception as e:
            print(f"Error listing devices: {e}")
        return

    # Set model path in environment if provided
    if args.model:
        os.environ['MODEL_PATH'] = args.model

    try:
        # Create and start monitor
        monitor = AudioMonitor(config_file=args.config)

        if args.test_telegram:
            # Test Telegram connection only
            monitor._initialize_components()
            print("Telegram connection test successful!")
            return

        # Start monitoring
        monitor.start()

    except KeyboardInterrupt:
        print("\nShutdown requested by user")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
