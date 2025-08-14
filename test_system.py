#!/usr/bin/env python3
"""
System test script to verify all components work correctly.
Windows-compatible version with ASCII characters only.
"""

import sys
import os
import time
import numpy as np
from datetime import datetime

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test that all modules can be imported without errors."""
    print("Testing imports...")
    
    try:
        from src.yamnet_classifier import YAMNetClassifier
        print("[OK] YAMNetClassifier imported successfully")
    except Exception as e:
        print(f"[FAIL] YAMNetClassifier import failed: {e}")
        return False
    
    try:
        from src.audio_capture import AudioCapture
        print("[OK] AudioCapture imported successfully")
    except Exception as e:
        print(f"[FAIL] AudioCapture import failed: {e}")
        return False
    
    try:
        from src.telegram_notifier import TelegramNotifier
        print("[OK] TelegramNotifier imported successfully")
    except Exception as e:
        print(f"[FAIL] TelegramNotifier import failed: {e}")
        return False
    
    try:
        from src.sound_detector import SoundDetector
        print("[OK] SoundDetector imported successfully")
    except Exception as e:
        print(f"[FAIL] SoundDetector import failed: {e}")
        return False
    
    try:
        from src.audio_monitor import AudioMonitor
        print("[OK] AudioMonitor imported successfully")
    except Exception as e:
        print(f"[FAIL] AudioMonitor import failed: {e}")
        return False
    
    return True

def test_yamnet_classifier():
    """Test YAMNet classifier functionality."""
    print("\nTesting YAMNet classifier...")
    
    try:
        from src.yamnet_classifier import YAMNetClassifier
        
        # Check if model file exists
        if not os.path.exists('1.tflite'):
            print("[FAIL] Model file '1.tflite' not found")
            return False
        
        # Initialize classifier
        classifier = YAMNetClassifier('1.tflite')
        print("[OK] YAMNet classifier initialized")
        
        # Test with dummy audio data
        dummy_audio = np.random.randn(15600).astype(np.float32) * 0.1
        class_names, confidence_scores = classifier.classify(dummy_audio)
        
        if len(class_names) > 0 and len(confidence_scores) > 0:
            print(f"[OK] Classification successful: {len(class_names)} predictions")
            print(f"  Top prediction: {class_names[0]} ({confidence_scores[0]:.3f})")
        else:
            print("[FAIL] Classification returned empty results")
            return False
        
        # Test available classes
        available_classes = classifier.get_available_classes()
        if len(available_classes) > 0:
            print(f"[OK] Available classes: {len(available_classes)} classes loaded")
        else:
            print("[FAIL] No available classes found")
            return False
        
        return True
        
    except Exception as e:
        print(f"[FAIL] YAMNet classifier test failed: {e}")
        return False

def test_audio_capture():
    """Test audio capture functionality."""
    print("\nTesting audio capture...")
    
    try:
        from src.audio_capture import AudioCapture
        
        # Initialize audio capture
        capture = AudioCapture()
        print("[OK] AudioCapture initialized")
        
        # Test audio level (should be 0.0 when not capturing)
        level = capture.get_audio_level()
        print(f"[OK] Audio level: {level:.4f}")
        
        # Test is_active (should be False when not capturing)
        active = capture.is_active()
        print(f"[OK] Audio active status: {active}")
        
        # Cleanup
        capture.cleanup()
        print("[OK] AudioCapture cleanup successful")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Audio capture test failed: {e}")
        return False

def test_telegram_notifier():
    """Test Telegram notifier functionality."""
    print("\nTesting Telegram notifier...")
    
    try:
        from src.telegram_notifier import TelegramNotifier
        from dotenv import load_dotenv
        
        # Load configuration
        load_dotenv('config.env')
        bot_token = os.getenv('TELEGRAM_BOT_TOKEN', '')
        chat_id = os.getenv('TELEGRAM_CHAT_ID', '')
        
        if not bot_token or not chat_id:
            print("[FAIL] Telegram credentials not found in config.env")
            return False
        
        # Initialize notifier
        notifier = TelegramNotifier(bot_token, chat_id)
        print("[OK] TelegramNotifier initialized")
        
        # Test connection
        if notifier.test_connection():
            print("[OK] Telegram connection test successful")
        else:
            print("[FAIL] Telegram connection test failed")
            return False
        
        # Test notification stats
        stats = notifier.get_notification_stats()
        print(f"[OK] Notification stats: {stats['total_sound_types']} sound types")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Telegram notifier test failed: {e}")
        return False

def test_sound_detector():
    """Test sound detector functionality."""
    print("\nTesting sound detector...")
    
    try:
        from src.yamnet_classifier import YAMNetClassifier
        from src.telegram_notifier import TelegramNotifier
        from src.sound_detector import SoundDetector
        from dotenv import load_dotenv
        
        # Load configuration
        load_dotenv('config.env')
        bot_token = os.getenv('TELEGRAM_BOT_TOKEN', '')
        chat_id = os.getenv('TELEGRAM_CHAT_ID', '')
        target_sounds = ['Baby cry, infant cry', 'Glass', 'Fire alarm', 'Smoke detector, smoke alarm']
        
        # Initialize components
        classifier = YAMNetClassifier('1.tflite')
        notifier = TelegramNotifier(bot_token, chat_id)
        detector = SoundDetector(classifier, notifier, target_sounds)
        print("[OK] SoundDetector initialized")
        
        # Test with dummy audio
        dummy_audio = np.random.randn(15600).astype(np.float32) * 0.1
        detection_event = detector.process_audio(dummy_audio)
        
        if detection_event is None:
            print("[OK] No detection on random noise (expected)")
        else:
            print(f"[OK] Detection event: {detection_event.sound_type}")
        
        # Test detection stats
        stats = detector.get_detection_stats()
        print(f"[OK] Detection stats: {stats['total_processed']} processed, {stats['total_detections']} detections")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Sound detector test failed: {e}")
        return False

def test_audio_monitor():
    """Test audio monitor functionality."""
    print("\nTesting audio monitor...")
    
    try:
        from src.audio_monitor import AudioMonitor
        
        # Initialize monitor
        monitor = AudioMonitor('config.env')
        print("[OK] AudioMonitor initialized")
        
        # Test status
        status = monitor.get_status()
        print(f"[OK] Monitor status: running={status['is_running']}")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Audio monitor test failed: {e}")
        return False

def test_configuration():
    """Test configuration loading."""
    print("\nTesting configuration...")
    
    try:
        from dotenv import load_dotenv
        
        # Check if config file exists
        if not os.path.exists('config.env'):
            print("[FAIL] Configuration file 'config.env' not found")
            return False
        
        # Load configuration
        load_dotenv('config.env')
        
        # Check required settings
        required_settings = [
            'TELEGRAM_BOT_TOKEN',
            'TELEGRAM_CHAT_ID',
            'TARGET_SOUNDS'
        ]
        
        for setting in required_settings:
            value = os.getenv(setting, '')
            if not value:
                print(f"[FAIL] Required setting '{setting}' not found or empty")
                return False
            print(f"[OK] {setting}: configured")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Configuration test failed: {e}")
        return False

def main():
    """Run all system tests."""
    print("=" * 50)
    print("SYSTEM TEST SUITE")
    print("=" * 50)
    
    tests = [
        ("Configuration", test_configuration),
        ("Imports", test_imports),
        ("YAMNet Classifier", test_yamnet_classifier),
        ("Audio Capture", test_audio_capture),
        ("Telegram Notifier", test_telegram_notifier),
        ("Sound Detector", test_sound_detector),
        ("Audio Monitor", test_audio_monitor),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if test_func():
                passed += 1
                print(f"[PASS] {test_name} PASSED")
            else:
                print(f"[FAIL] {test_name} FAILED")
        except Exception as e:
            print(f"[FAIL] {test_name} FAILED with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"TEST RESULTS: {passed}/{total} tests passed")
    print("=" * 50)
    
    if passed == total:
        print("SUCCESS: ALL TESTS PASSED! System is ready to use.")
        return 0
    else:
        print("ERROR: Some tests failed. Please check the errors above.")
        return 1

if __name__ == '__main__':
    sys.exit(main())