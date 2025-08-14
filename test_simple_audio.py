#!/usr/bin/env python3
"""
Simple audio test using WAV files from test_audio directory
"""

import sys
import os
import time
import platform
import numpy as np
import wave

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def load_simple_wav(file_path):
    """Simple WAV file loader."""
    try:
        with wave.open(file_path, 'rb') as wav_file:
            frames = wav_file.readframes(-1)
            sample_rate = wav_file.getframerate()
            channels = wav_file.getnchannels()
            
            # Convert to numpy array
            audio_data = np.frombuffer(frames, dtype=np.int16)
            
            # Convert to float32 and normalize
            audio_data = audio_data.astype(np.float32) / 32768.0
            
            # Convert to mono if stereo
            if channels > 1:
                audio_data = audio_data.reshape(-1, channels)
                audio_data = np.mean(audio_data, axis=1)
            
            # Resample to 16kHz if needed (simple)
            if sample_rate != 16000:
                target_length = int(len(audio_data) * 16000 / sample_rate)
                audio_data = np.interp(
                    np.linspace(0, len(audio_data), target_length),
                    np.arange(len(audio_data)),
                    audio_data
                )
            
            # Ensure exactly 15600 samples (0.975 seconds at 16kHz)
            target_length = 15600
            if len(audio_data) > target_length:
                start = (len(audio_data) - target_length) // 2
                audio_data = audio_data[start:start + target_length]
            elif len(audio_data) < target_length:
                padding = target_length - len(audio_data)
                audio_data = np.pad(audio_data, (0, padding), mode='constant')
            
            return audio_data.astype(np.float32)
            
    except Exception as e:
        print(f"❌ Failed to load {file_path}: {e}")
        return None

def test_audio_files():
    """Test audio files with YAMNet."""
    print("🎵 Simple Audio File Test")
    print("=" * 30)
    
    # System info
    system = platform.system()
    arch = platform.machine()
    print(f"Platform: {system} {arch}")
    
    # Find audio files
    test_audio_dir = "test_audio"
    if not os.path.exists(test_audio_dir):
        print(f"❌ Directory {test_audio_dir} not found")
        return
    
    wav_files = [f for f in os.listdir(test_audio_dir) if f.endswith('.wav')]
    print(f"Found {len(wav_files)} WAV files")
    
    if not wav_files:
        print("❌ No WAV files found")
        return
    
    # Test first few files
    test_files = wav_files[:5]  # Test first 5 files
    
    try:
        from src.yamnet_classifier import YAMNetClassifier
        
        print("\nLoading YAMNet model...")
        start_time = time.time()
        classifier = YAMNetClassifier("1.tflite")
        load_time = time.time() - start_time
        print(f"✓ Model loaded in {load_time:.2f}s")
        
        total_time = 0
        successful_tests = 0
        
        for filename in test_files:
            file_path = os.path.join(test_audio_dir, filename)
            print(f"\n🎵 Testing: {filename}")
            
            # Load audio
            audio_data = load_simple_wav(file_path)
            if audio_data is None:
                continue
            
            print(f"  Audio shape: {audio_data.shape}")
            print(f"  Audio range: [{audio_data.min():.3f}, {audio_data.max():.3f}]")
            
            # Classify
            start = time.time()
            classes, scores = classifier.classify(audio_data)
            end = time.time()
            
            classification_time = end - start
            total_time += classification_time
            successful_tests += 1
            
            print(f"  Classification time: {classification_time:.3f}s")
            
            if classes and scores:
                print(f"  Top prediction: {classes[0]} ({scores[0]:.3f})")
                
                # Check for target sounds
                target_sounds = ['Baby cry, infant cry', 'Glass', 'Fire alarm', 'Smoke detector, smoke alarm']
                for target in target_sounds:
                    for i, (class_name, score) in enumerate(zip(classes[:5], scores[:5])):
                        if target.lower() in class_name.lower() or class_name.lower() in target.lower():
                            print(f"  🎯 TARGET DETECTED: {class_name} ({score:.3f}) - matches {target}")
                            break
            else:
                print("  ❌ No predictions returned")
        
        # Summary
        if successful_tests > 0:
            avg_time = total_time / successful_tests
            fps = 1.0 / avg_time
            
            print(f"\n📊 Performance Summary:")
            print(f"  Files tested: {successful_tests}")
            print(f"  Total time: {total_time:.2f}s")
            print(f"  Average per file: {avg_time:.3f}s")
            print(f"  Processing speed: {fps:.1f} FPS")
            print(f"  Real-time capable: {'✓ Yes' if fps >= 10 else '❌ No'}")
            
            # Platform comparison
            if "arm" in arch.lower():
                print(f"\n🍓 Raspberry Pi Performance:")
                if fps >= 5:
                    print("  ✓ Good performance for Pi")
                elif fps >= 2:
                    print("  ⚠ Acceptable performance, consider optimization")
                else:
                    print("  ❌ Poor performance, needs optimization")
            else:
                print(f"\n💻 PC Performance:")
                if fps >= 15:
                    print("  ✓ Excellent performance")
                elif fps >= 10:
                    print("  ✓ Good performance")
                else:
                    print("  ⚠ Consider system optimization")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_audio_files()
