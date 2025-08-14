#!/usr/bin/env python3
"""
Real-time Audio Monitoring System with YAMNet
Main entry point for the application.
"""

import sys
import os

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.audio_monitor import main

if __name__ == '__main__':
    main()
