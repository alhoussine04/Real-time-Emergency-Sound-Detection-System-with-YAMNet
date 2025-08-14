# Raspberry Pi Installation Guide

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
