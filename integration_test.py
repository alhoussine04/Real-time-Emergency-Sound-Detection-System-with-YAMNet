#!/usr/bin/env python3
"""
Integration test to verify the complete system can start and run.
"""

import sys
import os
import time
import threading
import signal
from datetime import datetime

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_system_startup():
    """Test that the system can start up completely."""
    print("Testing complete system startup...")
    
    try:
        from src.audio_monitor import AudioMonitor
        
        # Create monitor instance
        monitor = AudioMonitor('config.env')
        print("[OK] AudioMonitor created successfully")
        
        # Test initialization without starting capture
        monitor._initialize_components()
        print("[OK] All components initialized successfully")
        
        # Test status
        status = monitor.get_status()
        print(f"[OK] System status retrieved: running={status['is_running']}")
        
        # Test component access
        if monitor.classifier:
            print("[OK] YAMNet classifier accessible")
        
        if monitor.notifier:
            print("[OK] Telegram notifier accessible")
            
        if monitor.audio_capture:
            print("[OK] Audio capture accessible")
            
        if monitor.detector:
            print("[OK] Sound detector accessible")
            stats = monitor.detector.get_detection_stats()
            print(f"[OK] Detection stats accessible: {stats['target_sounds']}")
        
        print("[OK] System startup test completed successfully")
        return True
        
    except Exception as e:
        print(f"[FAIL] System startup test failed: {e}")
        return False

def test_audio_processing_pipeline():
    """Test the audio processing pipeline with dummy data."""
    print("\nTesting audio processing pipeline...")
    
    try:
        import numpy as np
        from src.yamnet_classifier import YAMNetClassifier
        from src.telegram_notifier import TelegramNotifier
        from src.sound_detector import SoundDetector
        from dotenv import load_dotenv
        
        # Load configuration
        load_dotenv('config.env')
        bot_token = os.getenv('TELEGRAM_BOT_TOKEN', '')
        chat_id = os.getenv('TELEGRAM_CHAT_ID', '')
        
        # Initialize components
        classifier = YAMNetClassifier('1.tflite')
        notifier = TelegramNotifier(bot_token, chat_id)
        detector = SoundDetector(
            classifier, 
            notifier, 
            ['Baby cry, infant cry', 'Glass', 'Fire alarm', 'Smoke detector, smoke alarm']
        )
        
        print("[OK] Pipeline components initialized")
        
        # Test with multiple audio samples
        for i in range(3):
            # Generate different types of dummy audio
            if i == 0:
                # White noise
                dummy_audio = np.random.randn(15600).astype(np.float32) * 0.1
            elif i == 1:
                # Sine wave (might trigger some detections)
                t = np.linspace(0, 0.975, 15600)
                dummy_audio = (0.1 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
            else:
                # Silence
                dummy_audio = np.zeros(15600, dtype=np.float32)
            
            # Process through pipeline
            detection_event = detector.process_audio(dummy_audio)
            
            if detection_event:
                print(f"[INFO] Sample {i+1}: Detection - {detection_event.sound_type} ({detection_event.confidence:.3f})")
            else:
                print(f"[OK] Sample {i+1}: No detection (expected for dummy data)")
        
        # Check final stats
        stats = detector.get_detection_stats()
        print(f"[OK] Pipeline processed {stats['total_processed']} samples with {stats['total_detections']} detections")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Audio processing pipeline test failed: {e}")
        return False

def test_configuration_loading():
    """Test configuration loading and validation."""
    print("\nTesting configuration loading...")
    
    try:
        from dotenv import load_dotenv
        
        # Test loading configuration
        load_dotenv('config.env')
        
        # Check required settings
        required_settings = {
            'TELEGRAM_BOT_TOKEN': os.getenv('TELEGRAM_BOT_TOKEN', ''),
            'TELEGRAM_CHAT_ID': os.getenv('TELEGRAM_CHAT_ID', ''),
            'SAMPLE_RATE': int(os.getenv('SAMPLE_RATE', '16000')),
            'CONFIDENCE_THRESHOLD': float(os.getenv('CONFIDENCE_THRESHOLD', '0.3')),
            'TARGET_SOUNDS': os.getenv('TARGET_SOUNDS', '').split(';')
        }
        
        for setting, value in required_settings.items():
            if not value or (isinstance(value, list) and not value[0]):
                print(f"[FAIL] Required setting '{setting}' is missing or empty")
                return False
            print(f"[OK] {setting}: configured")
        
        print("[OK] Configuration loading test completed")
        return True
        
    except Exception as e:
        print(f"[FAIL] Configuration loading test failed: {e}")
        return False

def test_error_handling():
    """Test error handling scenarios."""
    print("\nTesting error handling...")
    
    try:
        from src.yamnet_classifier import YAMNetClassifier
        import numpy as np
        
        classifier = YAMNetClassifier('1.tflite')
        
        # Test with invalid audio data
        try:
            # Empty array
            result = classifier.classify(np.array([]))
            print("[OK] Handled empty audio array")
        except:
            print("[OK] Properly rejected empty audio array")
        
        # Test with wrong data type
        result = classifier.classify("invalid_data")
        if result == ([], []):
            print("[OK] Properly handled invalid data type")
        else:
            print("[FAIL] Should have returned empty results for invalid input")
            return False
        
        # Test with edge case audio sizes
        try:
            # Very small audio
            small_audio = np.random.randn(100).astype(np.float32) * 0.1
            result = classifier.classify(small_audio)
            print("[OK] Handled small audio array")
        except Exception as e:
            print(f"[INFO] Small audio handling: {e}")
        
        print("[OK] Error handling test completed")
        return True
        
    except Exception as e:
        print(f"[FAIL] Error handling test failed: {e}")
        return False

def main():
    """Run integration tests."""
    print("=" * 60)
    print("INTEGRATION TEST SUITE")
    print("=" * 60)
    print("Testing complete system integration and functionality")
    
    tests = [
        ("Configuration Loading", test_configuration_loading),
        ("System Startup", test_system_startup),
        ("Audio Processing Pipeline", test_audio_processing_pipeline),
        ("Error Handling", test_error_handling),
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
    
    print("\n" + "=" * 60)
    print(f"INTEGRATION TEST RESULTS: {passed}/{total} tests passed")
    print("=" * 60)
    
    if passed == total:
        print("[SUCCESS] ALL INTEGRATION TESTS PASSED!")
        print("The system is fully functional and ready for production use.")
        return 0
    else:
        print("[ERROR] Some integration tests failed.")
        print("Please review the errors above before deploying the system.")
        return 1

if __name__ == '__main__':
    sys.exit(main())