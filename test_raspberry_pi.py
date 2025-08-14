#!/usr/bin/env python3
"""
Raspberry Pi compatibility test script.
Tests the system specifically for Raspberry Pi deployment.
"""

import sys
import os
import platform
import subprocess
import time

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def check_raspberry_pi():
    """Check if running on Raspberry Pi."""
    print("Checking Raspberry Pi compatibility...")
    
    # Check if running on ARM architecture
    machine = platform.machine()
    system = platform.system()
    
    print(f"System: {system}")
    print(f"Architecture: {machine}")
    print(f"Platform: {platform.platform()}")
    
    is_rpi = False
    if machine.startswith('arm') or machine.startswith('aarch'):
        print("[INFO] ARM architecture detected - likely Raspberry Pi")
        is_rpi = True
    
    # Check for Raspberry Pi specific files
    rpi_files = ['/proc/device-tree/model', '/sys/firmware/devicetree/base/model']
    for rpi_file in rpi_files:
        if os.path.exists(rpi_file):
            try:
                with open(rpi_file, 'r') as f:
                    model = f.read().strip()
                    if 'Raspberry Pi' in model:
                        print(f"[OK] Raspberry Pi detected: {model}")
                        is_rpi = True
                        break
            except:
                pass
    
    if not is_rpi:
        print("[INFO] Not running on Raspberry Pi - testing compatibility anyway")
    
    return is_rpi

def test_tflite_runtime():
    """Test TensorFlow Lite Runtime availability."""
    print("\nTesting TensorFlow Lite Runtime...")
    
    try:
        import tflite_runtime.interpreter as tflite
        print("[OK] TensorFlow Lite Runtime available")
        
        # Test basic functionality
        try:
            # This will fail without a model, but tests import
            interpreter = tflite.Interpreter(model_path="nonexistent.tflite")
        except Exception as e:
            if "nonexistent.tflite" in str(e):
                print("[OK] TensorFlow Lite Runtime functional")
            else:
                print(f"[WARNING] TensorFlow Lite Runtime issue: {e}")
        
        return True
        
    except ImportError:
        print("[INFO] TensorFlow Lite Runtime not available")
        
        # Try full TensorFlow as fallback
        try:
            import tensorflow as tf
            print(f"[OK] Full TensorFlow available: {tf.__version__}")
            return True
        except ImportError:
            print("[ERROR] Neither TensorFlow Lite Runtime nor TensorFlow available")
            return False

def test_audio_on_rpi():
    """Test audio functionality on Raspberry Pi."""
    print("\nTesting audio functionality...")
    
    try:
        import pyaudio
        
        # Initialize PyAudio
        audio = pyaudio.PyAudio()
        device_count = audio.get_device_count()
        
        print(f"Found {device_count} audio devices:")
        
        input_devices = []
        for i in range(device_count):
            try:
                device_info = audio.get_device_info_by_index(i)
                if device_info['maxInputChannels'] > 0:
                    input_devices.append((i, device_info['name']))
                    print(f"  Device {i}: {device_info['name']} "
                          f"(Channels: {device_info['maxInputChannels']}, "
                          f"Sample Rate: {device_info['defaultSampleRate']})")
            except:
                continue
        
        audio.terminate()
        
        if input_devices:
            print(f"[OK] Found {len(input_devices)} input devices")
            
            # Check for common Raspberry Pi audio devices
            rpi_audio_devices = ['USB Audio', 'bcm2835', 'ALSA', 'pulse']
            for device_id, device_name in input_devices:
                for rpi_device in rpi_audio_devices:
                    if rpi_device.lower() in device_name.lower():
                        print(f"[OK] Raspberry Pi audio device detected: {device_name}")
                        break
            
            return True
        else:
            print("[WARNING] No audio input devices found")
            print("Make sure you have a USB microphone or audio HAT connected")
            return False
            
    except ImportError:
        print("[ERROR] PyAudio not available")
        return False
    except Exception as e:
        print(f"[ERROR] Audio test failed: {e}")
        return False

def test_yamnet_on_rpi():
    """Test YAMNet classifier on Raspberry Pi."""
    print("\nTesting YAMNet classifier...")
    
    try:
        from src.yamnet_classifier import YAMNetClassifier
        import numpy as np
        
        # Check if model file exists
        if not os.path.exists('1.tflite'):
            print("[ERROR] Model file '1.tflite' not found")
            return False
        
        # Initialize classifier
        start_time = time.time()
        classifier = YAMNetClassifier('1.tflite')
        load_time = time.time() - start_time
        
        print(f"[OK] YAMNet classifier loaded in {load_time:.2f} seconds")
        
        # Test classification performance
        dummy_audio = np.random.randn(15600).astype(np.float32) * 0.1
        
        # Warm-up run
        classifier.classify(dummy_audio)
        
        # Performance test
        num_tests = 5
        total_time = 0
        
        for i in range(num_tests):
            start_time = time.time()
            class_names, confidence_scores = classifier.classify(dummy_audio)
            inference_time = time.time() - start_time
            total_time += inference_time
            
            if i == 0:  # Print first result
                print(f"[OK] Classification successful: {len(class_names)} predictions")
                if class_names:
                    print(f"  Top prediction: {class_names[0]} ({confidence_scores[0]:.3f})")
        
        avg_time = total_time / num_tests
        fps = 1.0 / avg_time
        
        print(f"[OK] Average inference time: {avg_time*1000:.1f}ms")
        print(f"[OK] Processing rate: {fps:.1f} FPS")
        
        # Performance recommendations for Raspberry Pi
        if avg_time > 0.1:  # 100ms
            print("[WARNING] Inference time is high for real-time processing")
            print("Consider using a faster Raspberry Pi model or optimizing the model")
        elif avg_time > 0.05:  # 50ms
            print("[INFO] Inference time is acceptable but could be improved")
        else:
            print("[OK] Excellent inference performance for Raspberry Pi")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] YAMNet test failed: {e}")
        return False

def test_memory_usage():
    """Test memory usage on Raspberry Pi."""
    print("\nTesting memory usage...")
    
    try:
        import psutil
        
        # Get initial memory usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        print(f"Initial memory usage: {initial_memory:.1f} MB")
        
        # Load the system components
        from src.yamnet_classifier import YAMNetClassifier
        from src.telegram_notifier import TelegramNotifier
        from src.audio_capture import AudioCapture
        from dotenv import load_dotenv
        
        load_dotenv('config.env')
        
        # Initialize components
        if os.path.exists('1.tflite'):
            classifier = YAMNetClassifier('1.tflite')
            after_model_memory = process.memory_info().rss / 1024 / 1024
            print(f"Memory after model load: {after_model_memory:.1f} MB")
            print(f"Model memory usage: {after_model_memory - initial_memory:.1f} MB")
        
        bot_token = os.getenv('TELEGRAM_BOT_TOKEN', 'dummy_token')
        chat_id = os.getenv('TELEGRAM_CHAT_ID', 'dummy_chat')
        
        if bot_token != 'dummy_token':
            notifier = TelegramNotifier(bot_token, chat_id)
        
        audio_capture = AudioCapture()
        
        final_memory = process.memory_info().rss / 1024 / 1024
        total_usage = final_memory - initial_memory
        
        print(f"Final memory usage: {final_memory:.1f} MB")
        print(f"Total system memory usage: {total_usage:.1f} MB")
        
        # Memory recommendations for Raspberry Pi
        if total_usage > 500:  # 500MB
            print("[WARNING] High memory usage - may cause issues on Raspberry Pi Zero")
        elif total_usage > 200:  # 200MB
            print("[INFO] Moderate memory usage - should work on Raspberry Pi 3+")
        else:
            print("[OK] Low memory usage - suitable for all Raspberry Pi models")
        
        # Cleanup
        audio_capture.cleanup()
        
        return True
        
    except ImportError:
        print("[INFO] psutil not available - skipping memory test")
        return True
    except Exception as e:
        print(f"[ERROR] Memory test failed: {e}")
        return False

def test_system_integration_rpi():
    """Test complete system integration on Raspberry Pi."""
    print("\nTesting system integration...")
    
    try:
        from src.audio_monitor import AudioMonitor
        
        # Create monitor instance
        monitor = AudioMonitor('config.env')
        print("[OK] AudioMonitor created successfully")
        
        # Test component initialization
        monitor._initialize_components()
        print("[OK] All components initialized successfully")
        
        # Test status
        status = monitor.get_status()
        print(f"[OK] System status: running={status['is_running']}")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] System integration test failed: {e}")
        return False

def create_rpi_installation_guide():
    """Create installation guide for Raspberry Pi."""
    print("\nCreating Raspberry Pi installation guide...")
    
    guide = """# Raspberry Pi Installation Guide

## Prerequisites
- Raspberry Pi 3B+ or newer (Raspberry Pi 4 recommended)
- MicroSD card (16GB or larger)
- USB microphone or audio HAT
- Internet connection

## System Setup

### 1. Install Raspberry Pi OS
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install required system packages
sudo apt install -y python3-pip python3-venv git portaudio19-dev python3-pyaudio
```

### 2. Install Python Dependencies
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Raspberry Pi optimized packages
pip install -r requirements-rpi.txt
```

### 3. Audio Configuration
```bash
# Test audio devices
arecord -l

# Configure default audio device if needed
sudo nano /etc/asound.conf
```

### 4. Enable Audio
```bash
# Add user to audio group
sudo usermod -a -G audio $USER

# Reboot to apply changes
sudo reboot
```

### 5. Test Installation
```bash
python test_raspberry_pi.py
```

## Performance Optimization

### For Raspberry Pi 3/3B+:
- Use confidence threshold >= 0.4 to reduce false positives
- Increase detection cooldown to 10+ seconds
- Consider using chunk size of 2048 for better performance

### For Raspberry Pi 4:
- Default settings should work well
- Can handle real-time processing at 16kHz

### For Raspberry Pi Zero:
- Not recommended due to limited processing power
- If using, set confidence threshold to 0.5+ and cooldown to 30+ seconds

## Troubleshooting

### Audio Issues:
```bash
# Check audio devices
python main.py --list-devices

# Test microphone
arecord -d 5 test.wav && aplay test.wav
```

### Memory Issues:
```bash
# Check memory usage
free -h

# Increase swap if needed
sudo dphys-swapfile swapoff
sudo nano /etc/dphys-swapfile  # Set CONF_SWAPSIZE=1024
sudo dphys-swapfile setup
sudo dphys-swapfile swapon
```

### Performance Issues:
- Reduce model precision if available
- Increase processing intervals
- Use hardware acceleration if available

## Autostart Setup
```bash
# Create systemd service
sudo nano /etc/systemd/system/audio-monitor.service

# Enable service
sudo systemctl enable audio-monitor.service
sudo systemctl start audio-monitor.service
```
"""
    
    try:
        with open('RASPBERRY_PI_GUIDE.md', 'w') as f:
            f.write(guide)
        print("[OK] Raspberry Pi installation guide created: RASPBERRY_PI_GUIDE.md")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to create installation guide: {e}")
        return False

def main():
    """Run Raspberry Pi compatibility tests."""
    print("=" * 60)
    print("RASPBERRY PI COMPATIBILITY TEST")
    print("=" * 60)
    
    tests = [
        ("Raspberry Pi Detection", check_raspberry_pi),
        ("TensorFlow Lite Runtime", test_tflite_runtime),
        ("Audio Functionality", test_audio_on_rpi),
        ("YAMNet Performance", test_yamnet_on_rpi),
        ("Memory Usage", test_memory_usage),
        ("System Integration", test_system_integration_rpi),
        ("Installation Guide", create_rpi_installation_guide),
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
    print(f"RASPBERRY PI TEST RESULTS: {passed}/{total} tests passed")
    print("=" * 60)
    
    if passed >= total - 1:  # Allow one test to fail
        print("[SUCCESS] System is compatible with Raspberry Pi!")
        print("\nRecommendations:")
        print("- Use requirements-rpi.txt for installation")
        print("- Follow RASPBERRY_PI_GUIDE.md for setup")
        print("- Test on your specific Raspberry Pi model")
        return 0
    else:
        print("[WARNING] Some compatibility issues found.")
        print("Please review the failed tests above.")
        return 1

if __name__ == '__main__':
    sys.exit(main())