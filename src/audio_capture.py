"""
Real-time audio capture system for the YAMNet audio monitoring.
"""

import pyaudio
import numpy as np
import threading
import queue
import logging
import time
from typing import Optional, Callable
from collections import deque


class AudioCapture:
    """
    Real-time audio capture system with buffering for YAMNet processing.
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        chunk_size: int = 1024,
        channels: int = 1,
        device_index: Optional[int] = None,
        buffer_duration: float = 0.975
    ):
        """
        Initialize the audio capture system.
        
        Args:
            sample_rate: Audio sample rate in Hz (YAMNet requires 16kHz)
            chunk_size: Number of samples per audio chunk
            channels: Number of audio channels (YAMNet requires mono)
            device_index: Audio device index (-1 for default)
            buffer_duration: Duration of audio buffer in seconds (YAMNet requires 0.975s)
        """
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.channels = channels
        self.device_index = device_index if device_index != -1 else None
        self.buffer_duration = buffer_duration
        
        # Calculate buffer size in samples
        self.buffer_size = int(sample_rate * buffer_duration)
        
        # Audio components
        self.audio = None
        self.stream = None
        
        # Threading and buffering
        self.audio_queue = queue.Queue(maxsize=100)
        self.audio_buffer = deque(maxlen=self.buffer_size)
        self.capture_thread = None
        self.is_capturing = False
        
        # Callback for processed audio
        self.audio_callback: Optional[Callable[[np.ndarray], None]] = None
        
        self.logger = logging.getLogger(__name__)
        
        self._initialize_audio()
    
    def _initialize_audio(self) -> None:
        """Initialize PyAudio and list available devices."""
        try:
            self.audio = pyaudio.PyAudio()
            
            # Log available audio devices
            self.logger.info("Available audio devices:")
            for i in range(self.audio.get_device_count()):
                device_info = self.audio.get_device_info_by_index(i)
                if device_info['maxInputChannels'] > 0:
                    self.logger.info(
                        f"  Device {i}: {device_info['name']} "
                        f"(Channels: {device_info['maxInputChannels']}, "
                        f"Sample Rate: {device_info['defaultSampleRate']})"
                    )
            
            # Get default device if none specified
            if self.device_index is None:
                default_device = self.audio.get_default_input_device_info()
                self.device_index = default_device['index']
                self.logger.info(f"Using default input device: {default_device['name']}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize audio: {e}")
            raise
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Callback function for audio stream."""
        if status:
            self.logger.warning(f"Audio callback status: {status}")
        
        try:
            # Convert bytes to numpy array
            audio_data = np.frombuffer(in_data, dtype=np.int16)
            
            # Convert to float32 and normalize to [-1.0, 1.0]
            audio_data = audio_data.astype(np.float32) / 32768.0
            
            # Add to queue for processing
            if not self.audio_queue.full():
                self.audio_queue.put(audio_data)
            else:
                self.logger.warning("Audio queue is full, dropping frame")
                
        except Exception as e:
            self.logger.error(f"Error in audio callback: {e}")
        
        return (None, pyaudio.paContinue)
    
    def _process_audio_thread(self) -> None:
        """Thread function to process audio data."""
        while self.is_capturing:
            try:
                # Get audio chunk from queue (with timeout)
                audio_chunk = self.audio_queue.get(timeout=1.0)
                
                # Add to rolling buffer
                self.audio_buffer.extend(audio_chunk)
                
                # If buffer is full, process it
                if len(self.audio_buffer) >= self.buffer_size:
                    # Get the latest buffer_size samples
                    buffer_array = np.array(list(self.audio_buffer)[-self.buffer_size:])
                    
                    # Call the audio callback if set
                    if self.audio_callback:
                        self.audio_callback(buffer_array)
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Error processing audio: {e}")
    
    def start_capture(self, audio_callback: Callable[[np.ndarray], None]) -> None:
        """
        Start audio capture.
        
        Args:
            audio_callback: Function to call with processed audio data
        """
        if self.is_capturing:
            self.logger.warning("Audio capture is already running")
            return
        
        self.audio_callback = audio_callback
        
        try:
            # Open audio stream
            self.stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=self.device_index,
                frames_per_buffer=self.chunk_size,
                stream_callback=self._audio_callback
            )
            
            # Start capture flag and thread
            self.is_capturing = True
            self.capture_thread = threading.Thread(
                target=self._process_audio_thread,
                daemon=True
            )
            self.capture_thread.start()
            
            # Start the audio stream
            self.stream.start_stream()
            
            self.logger.info("Audio capture started")
            
        except Exception as e:
            self.logger.error(f"Failed to start audio capture: {e}")
            self.is_capturing = False
            raise
    
    def stop_capture(self) -> None:
        """Stop audio capture."""
        if not self.is_capturing:
            return
        
        self.logger.info("Stopping audio capture...")
        
        # Stop capture flag
        self.is_capturing = False
        
        # Stop and close audio stream
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
        
        # Wait for thread to finish
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=2.0)
        
        self.logger.info("Audio capture stopped")
    
    def get_audio_level(self) -> float:
        """
        Get current audio level (RMS).
        
        Returns:
            Current audio level as RMS value
        """
        if len(self.audio_buffer) == 0:
            return 0.0
        
        # Calculate RMS of recent audio
        recent_samples = list(self.audio_buffer)[-1000:]  # Last ~60ms at 16kHz
        if len(recent_samples) == 0:
            return 0.0
        
        rms = np.sqrt(np.mean(np.square(recent_samples)))
        return float(rms)
    
    def is_active(self) -> bool:
        """Check if audio capture is active."""
        return self.is_capturing and self.stream and self.stream.is_active()
    
    def cleanup(self) -> None:
        """Clean up audio resources."""
        self.stop_capture()
        
        if self.audio:
            self.audio.terminate()
            self.audio = None
        
        self.logger.info("Audio capture cleaned up")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()
