#!/usr/bin/env python3
"""
Setup script for the Real-time Audio Monitoring System.
This script helps users install dependencies and verify their system.
Windows-compatible version with ASCII characters only.
"""

import sys
import os
import subprocess
import platform
from pathlib import Path

def print_header(title):
    """Print a formatted header."""
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)

def print_step(step_num, description):
    """Print a formatted step."""
    print(f"\n[Step {step_num}] {description}")
    print("-" * 40)

def check_python_version():
    """Check if Python version is compatible."""
    print_step(1, "Checking Python Version")
    
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 7):
        print("[ERROR] Python 3.7 or higher is required")
        print("Please upgrade your Python installation")
        return False
    
    print("[OK] Python version is compatible")
    return True

def check_system_info():
    """Display system information."""
    print_step(2, "System Information")
    
    print(f"Operating System: {platform.system()} {platform.release()}")
    print(f"Architecture: {platform.machine()}")
    print(f"Platform: {platform.platform()}")
    
    return True

def install_dependencies():
    """Install required Python packages."""
    print_step(3, "Installing Dependencies")
    
    requirements_file = "requirements.txt"
    
    if not os.path.exists(requirements_file):
        print(f"[ERROR] {requirements_file} not found")
        return False
    
    print(f"Installing packages from {requirements_file}...")
    
    try:
        # Use pip to install requirements
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", requirements_file
        ], capture_output=True, text=True, check=True)
        
        print("[OK] Dependencies installed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed to install dependencies")
        print(f"Error output: {e.stderr}")
        return False
    except Exception as e:
        print(f"[ERROR] Unexpected error during installation: {e}")
        return False

def check_model_file():
    """Check if YAMNet model file exists."""
    print_step(4, "Checking Model File")
    
    model_file = "1.tflite"
    
    if os.path.exists(model_file):
        file_size = os.path.getsize(model_file)
        print(f"[OK] Model file found: {model_file} ({file_size:,} bytes)")
        return True
    else:
        print(f"[ERROR] Model file not found: {model_file}")
        print("Please ensure the YAMNet TF-Lite model file is in the project directory")
        return False

def check_config_file():
    """Check if configuration file exists."""
    print_step(5, "Checking Configuration")
    
    config_file = "config.env"
    
    if os.path.exists(config_file):
        print(f"[OK] Configuration file found: {config_file}")
        
        # Check if required settings are present
        try:
            with open(config_file, 'r') as f:
                content = f.read()
                
            required_settings = [
                'TELEGRAM_BOT_TOKEN',
                'TELEGRAM_CHAT_ID',
                'TARGET_SOUNDS'
            ]
            
            missing_settings = []
            for setting in required_settings:
                if setting not in content or f"{setting}=" not in content:
                    missing_settings.append(setting)
            
            if missing_settings:
                print(f"[WARNING] Missing configuration settings: {', '.join(missing_settings)}")
                print("Please update your config.env file with the required settings")
            else:
                print("[OK] All required configuration settings found")
            
            return True
            
        except Exception as e:
            print(f"[ERROR] Failed to read configuration file: {e}")
            return False
    else:
        print(f"[ERROR] Configuration file not found: {config_file}")
        print("Please create a config.env file with your settings")
        return False

def test_audio_devices():
    """Test audio device availability."""
    print_step(6, "Testing Audio Devices")
    
    try:
        import pyaudio
        
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
            return True
        else:
            print("[ERROR] No audio input devices found")
            print("Please connect a microphone or check your audio drivers")
            return False
            
    except ImportError:
        print("[ERROR] PyAudio not installed")
        return False
    except Exception as e:
        print(f"[ERROR] Failed to test audio devices: {e}")
        return False

def test_tensorflow():
    """Test TensorFlow installation."""
    print_step(7, "Testing TensorFlow")
    
    try:
        import tensorflow as tf
        print(f"TensorFlow version: {tf.__version__}")
        
        # Test TF-Lite interpreter
        try:
            interpreter = tf.lite.Interpreter(model_path="1.tflite")
            print("[OK] TensorFlow Lite interpreter working")
            return True
        except Exception as e:
            print(f"[ERROR] TensorFlow Lite test failed: {e}")
            return False
            
    except ImportError:
        print("[ERROR] TensorFlow not installed")
        return False
    except Exception as e:
        print(f"[ERROR] TensorFlow test failed: {e}")
        return False

def run_system_test():
    """Run the comprehensive system test."""
    print_step(8, "Running System Test")
    
    try:
        result = subprocess.run([
            sys.executable, "test_system.py"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("[OK] System test passed")
            return True
        else:
            print("[ERROR] System test failed")
            print("Error output:")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"[ERROR] Failed to run system test: {e}")
        return False

def create_sample_config():
    """Create a sample configuration file."""
    print_step("BONUS", "Creating Sample Configuration")
    
    sample_config = """# Telegram Bot Configuration
TELEGRAM_BOT_TOKEN=your_bot_token_here
TELEGRAM_CHAT_ID=your_chat_id_here

# Audio Configuration
SAMPLE_RATE=16000
CHUNK_SIZE=1024
CHANNELS=1
AUDIO_DEVICE_INDEX=-1

# Detection Configuration
CONFIDENCE_THRESHOLD=0.3
DETECTION_COOLDOWN=5.0
BUFFER_DURATION=0.975

# Target Sound Classes (YAMNet class names)
TARGET_SOUNDS=Baby cry, infant cry;Glass;Fire alarm;Smoke detector, smoke alarm

# Logging Configuration
LOG_LEVEL=INFO
LOG_FILE=audio_monitor.log
"""
    
    config_file = "config.env.sample"
    
    try:
        with open(config_file, 'w') as f:
            f.write(sample_config)
        
        print(f"[OK] Sample configuration created: {config_file}")
        print("Copy this file to config.env and update with your settings")
        return True
        
    except Exception as e:
        print(f"[ERROR] Failed to create sample configuration: {e}")
        return False

def main():
    """Main setup function."""
    print_header("Real-time Audio Monitoring System - Setup")
    print("This script will help you set up and verify your system.")
    
    steps = [
        ("Python Version", check_python_version),
        ("System Info", check_system_info),
        ("Dependencies", install_dependencies),
        ("Model File", check_model_file),
        ("Configuration", check_config_file),
        ("Audio Devices", test_audio_devices),
        ("TensorFlow", test_tensorflow),
        ("System Test", run_system_test),
    ]
    
    passed = 0
    total = len(steps)
    
    for step_name, step_func in steps:
        try:
            if step_func():
                passed += 1
            else:
                print(f"\n[WARNING] {step_name} check failed")
        except Exception as e:
            print(f"\n[ERROR] {step_name} check failed with exception: {e}")
    
    # Always try to create sample config
    create_sample_config()
    
    print_header("Setup Results")
    print(f"Completed: {passed}/{total} checks passed")
    
    if passed == total:
        print("\n[SUCCESS] SETUP COMPLETE! Your system is ready to use.")
        print("\nNext steps:")
        print("1. Update config.env with your Telegram bot credentials")
        print("2. Run: python main.py --test-telegram")
        print("3. Run: python main.py")
    else:
        print(f"\n[WARNING] Setup incomplete. Please fix the issues above.")
        print("Run this script again after resolving the problems.")
    
    return 0 if passed == total else 1

if __name__ == '__main__':
    sys.exit(main())