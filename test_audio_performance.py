#!/usr/bin/env python3
"""
Audio Performance Test - Compare Windows PC vs Raspberry Pi
Tests audio processing speed, accuracy, and resource usage.
"""

import sys
import os
import time
import psutil
import platform
import numpy as np
from datetime import datetime
import json

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def get_system_info():
    """Get detailed system information."""
    return {
        'platform': platform.system(),
        'platform_release': platform.release(),
        'platform_version': platform.version(),
        'architecture': platform.machine(),
        'processor': platform.processor(),
        'cpu_count': psutil.cpu_count(),
        'cpu_count_logical': psutil.cpu_count(logical=True),
        'memory_total': psutil.virtual_memory().total,
        'python_version': platform.python_version(),
    }

def test_yamnet_performance(config_file="config.env", num_tests=10):
    """Test YAMNet classification performance."""
    print(f"🧠 Testing YAMNet Performance ({num_tests} iterations)")
    print("=" * 50)
    
    try:
        from src.yamnet_classifier import YAMNetClassifier
        
        # Initialize classifier
        start_time = time.time()
        classifier = YAMNetClassifier("1.tflite")
        init_time = time.time() - start_time
        
        print(f"✓ Model initialization: {init_time:.3f}s")
        
        # Test different audio scenarios
        test_scenarios = [
            ("Silence", np.zeros(15600, dtype=np.float32)),
            ("White Noise", np.random.normal(0, 0.1, 15600).astype(np.float32)),
            ("Sine Wave 440Hz", np.sin(2 * np.pi * 440 * np.linspace(0, 0.975, 15600)).astype(np.float32)),
            ("Chirp", np.sin(2 * np.pi * np.linspace(100, 1000, 15600) * np.linspace(0, 0.975, 15600)).astype(np.float32))
        ]
        
        results = {}
        
        for scenario_name, audio_data in test_scenarios:
            print(f"\n📊 Testing: {scenario_name}")
            
            times = []
            predictions = []
            
            # Warm up
            classifier.classify(audio_data)
            
            # Performance test
            for i in range(num_tests):
                start = time.time()
                classes, scores = classifier.classify(audio_data)
                end = time.time()
                
                times.append(end - start)
                if classes:
                    predictions.append((classes[0], scores[0]))
                
                if (i + 1) % 5 == 0:
                    print(f"  Progress: {i + 1}/{num_tests}")
            
            # Calculate statistics
            avg_time = np.mean(times)
            min_time = np.min(times)
            max_time = np.max(times)
            std_time = np.std(times)
            
            results[scenario_name] = {
                'avg_time': avg_time,
                'min_time': min_time,
                'max_time': max_time,
                'std_time': std_time,
                'fps': 1.0 / avg_time,
                'predictions': predictions[:3]  # Top 3 predictions
            }
            
            print(f"  Average time: {avg_time:.3f}s ({1.0/avg_time:.1f} FPS)")
            print(f"  Min/Max time: {min_time:.3f}s / {max_time:.3f}s")
            print(f"  Std deviation: {std_time:.3f}s")
            if predictions:
                print(f"  Top prediction: {predictions[0][0]} ({predictions[0][1]:.3f})")
        
        return {
            'init_time': init_time,
            'scenarios': results
        }
        
    except Exception as e:
        print(f"❌ YAMNet test failed: {e}")
        return None

def test_audio_capture_performance(config_file="config.env", duration=10):
    """Test audio capture performance."""
    print(f"\n🎤 Testing Audio Capture Performance ({duration}s)")
    print("=" * 50)
    
    try:
        from src.audio_capture import AudioCapture
        from dotenv import load_dotenv
        
        # Load configuration
        load_dotenv(config_file)
        
        sample_rate = int(os.getenv('SAMPLE_RATE', '16000'))
        chunk_size = int(os.getenv('CHUNK_SIZE', '1024'))
        
        # Test audio capture
        audio_stats = {
            'chunks_received': 0,
            'total_samples': 0,
            'audio_levels': [],
            'timestamps': []
        }
        
        def audio_callback(audio_data):
            audio_stats['chunks_received'] += 1
            audio_stats['total_samples'] += len(audio_data)
            audio_stats['audio_levels'].append(np.sqrt(np.mean(np.square(audio_data))))
            audio_stats['timestamps'].append(time.time())
        
        with AudioCapture(
            sample_rate=sample_rate,
            chunk_size=chunk_size,
            channels=1
        ) as capture:
            
            print(f"✓ Audio capture initialized")
            print(f"  Sample rate: {sample_rate} Hz")
            print(f"  Chunk size: {chunk_size} samples")
            
            # Start capture
            start_time = time.time()
            capture.start_capture(audio_callback)
            
            # Monitor for specified duration
            while time.time() - start_time < duration:
                time.sleep(0.1)
                if audio_stats['chunks_received'] % 100 == 0 and audio_stats['chunks_received'] > 0:
                    elapsed = time.time() - start_time
                    rate = audio_stats['chunks_received'] / elapsed
                    print(f"  Progress: {elapsed:.1f}s, {audio_stats['chunks_received']} chunks ({rate:.1f} chunks/s)")
            
            capture.stop_capture()
        
        # Calculate statistics
        total_time = audio_stats['timestamps'][-1] - audio_stats['timestamps'][0] if audio_stats['timestamps'] else 0
        expected_chunks = int(total_time * sample_rate / chunk_size)
        chunk_rate = audio_stats['chunks_received'] / total_time if total_time > 0 else 0
        
        # Audio level statistics
        if audio_stats['audio_levels']:
            avg_level = np.mean(audio_stats['audio_levels'])
            max_level = np.max(audio_stats['audio_levels'])
            min_level = np.min(audio_stats['audio_levels'])
        else:
            avg_level = max_level = min_level = 0
        
        results = {
            'duration': total_time,
            'chunks_received': audio_stats['chunks_received'],
            'expected_chunks': expected_chunks,
            'chunk_rate': chunk_rate,
            'samples_per_second': audio_stats['total_samples'] / total_time if total_time > 0 else 0,
            'avg_audio_level': avg_level,
            'max_audio_level': max_level,
            'min_audio_level': min_level,
            'dropout_rate': max(0, (expected_chunks - audio_stats['chunks_received']) / expected_chunks) if expected_chunks > 0 else 0
        }
        
        print(f"\n📊 Audio Capture Results:")
        print(f"  Duration: {total_time:.1f}s")
        print(f"  Chunks received: {audio_stats['chunks_received']} / {expected_chunks} expected")
        print(f"  Chunk rate: {chunk_rate:.1f} chunks/s")
        print(f"  Dropout rate: {results['dropout_rate']*100:.1f}%")
        print(f"  Audio level: avg={avg_level:.4f}, max={max_level:.4f}")
        
        return results
        
    except Exception as e:
        print(f"❌ Audio capture test failed: {e}")
        return None

def test_system_resources(duration=30):
    """Monitor system resource usage during testing."""
    print(f"\n💻 Monitoring System Resources ({duration}s)")
    print("=" * 50)
    
    cpu_samples = []
    memory_samples = []
    timestamps = []
    
    start_time = time.time()
    
    while time.time() - start_time < duration:
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        cpu_samples.append(cpu_percent)
        memory_samples.append(memory.percent)
        timestamps.append(time.time() - start_time)
        
        print(f"  {timestamps[-1]:.0f}s: CPU={cpu_percent:.1f}%, Memory={memory.percent:.1f}%")
    
    results = {
        'duration': duration,
        'cpu_avg': np.mean(cpu_samples),
        'cpu_max': np.max(cpu_samples),
        'cpu_min': np.min(cpu_samples),
        'memory_avg': np.mean(memory_samples),
        'memory_max': np.max(memory_samples),
        'memory_min': np.min(memory_samples),
        'samples': len(cpu_samples)
    }
    
    print(f"\n📊 Resource Usage Results:")
    print(f"  CPU: avg={results['cpu_avg']:.1f}%, max={results['cpu_max']:.1f}%")
    print(f"  Memory: avg={results['memory_avg']:.1f}%, max={results['memory_max']:.1f}%")
    
    return results

def test_integrated_performance(config_file="config.env", duration=60):
    """Test integrated system performance (YAMNet + Audio Capture)."""
    print(f"\n🔄 Testing Integrated Performance ({duration}s)")
    print("=" * 50)
    
    try:
        from src.yamnet_classifier import YAMNetClassifier
        from src.audio_capture import AudioCapture
        from dotenv import load_dotenv
        
        # Load configuration
        load_dotenv(config_file)
        
        # Initialize components
        classifier = YAMNetClassifier("1.tflite")
        
        # Performance tracking
        stats = {
            'audio_chunks': 0,
            'classifications': 0,
            'classification_times': [],
            'detections': [],
            'start_time': time.time()
        }
        
        def process_audio(audio_data):
            stats['audio_chunks'] += 1
            
            # Classify audio
            start = time.time()
            classes, scores = classifier.classify(audio_data)
            end = time.time()
            
            stats['classifications'] += 1
            stats['classification_times'].append(end - start)
            
            # Check for interesting detections (confidence > 0.1)
            if classes and scores[0] > 0.1:
                stats['detections'].append({
                    'class': classes[0],
                    'confidence': scores[0],
                    'timestamp': time.time() - stats['start_time']
                })
            
            # Progress update
            if stats['classifications'] % 50 == 0:
                elapsed = time.time() - stats['start_time']
                rate = stats['classifications'] / elapsed
                avg_time = np.mean(stats['classification_times'][-50:])
                print(f"  {elapsed:.0f}s: {stats['classifications']} classifications ({rate:.1f}/s, {avg_time:.3f}s avg)")
        
        # Start integrated test
        sample_rate = int(os.getenv('SAMPLE_RATE', '16000'))
        chunk_size = int(os.getenv('CHUNK_SIZE', '1024'))
        
        with AudioCapture(
            sample_rate=sample_rate,
            chunk_size=chunk_size,
            channels=1
        ) as capture:
            
            capture.start_capture(process_audio)
            
            # Run for specified duration
            time.sleep(duration)
            
            capture.stop_capture()
        
        # Calculate results
        total_time = time.time() - stats['start_time']
        
        results = {
            'duration': total_time,
            'audio_chunks': stats['audio_chunks'],
            'classifications': stats['classifications'],
            'classification_rate': stats['classifications'] / total_time,
            'avg_classification_time': np.mean(stats['classification_times']),
            'max_classification_time': np.max(stats['classification_times']),
            'min_classification_time': np.min(stats['classification_times']),
            'detections_count': len(stats['detections']),
            'detections': stats['detections'][:10]  # Top 10 detections
        }
        
        print(f"\n📊 Integrated Performance Results:")
        print(f"  Duration: {total_time:.1f}s")
        print(f"  Classifications: {stats['classifications']} ({results['classification_rate']:.1f}/s)")
        print(f"  Avg classification time: {results['avg_classification_time']:.3f}s")
        print(f"  Detections: {len(stats['detections'])}")
        
        if stats['detections']:
            print(f"  Top detections:")
            for det in stats['detections'][:5]:
                print(f"    {det['timestamp']:.1f}s: {det['class']} ({det['confidence']:.3f})")
        
        return results
        
    except Exception as e:
        print(f"❌ Integrated test failed: {e}")
        return None

def save_results(results, platform_name):
    """Save test results to JSON file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"audio_performance_{platform_name}_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to: {filename}")
    return filename

def compare_results(file1, file2):
    """Compare results from two different platforms."""
    try:
        with open(file1, 'r') as f:
            results1 = json.load(f)
        with open(file2, 'r') as f:
            results2 = json.load(f)
        
        print(f"\n🔍 Comparison: {file1} vs {file2}")
        print("=" * 60)
        
        # Compare YAMNet performance
        if 'yamnet' in results1 and 'yamnet' in results2:
            print("\n🧠 YAMNet Performance Comparison:")
            for scenario in results1['yamnet']['scenarios']:
                if scenario in results2['yamnet']['scenarios']:
                    time1 = results1['yamnet']['scenarios'][scenario]['avg_time']
                    time2 = results2['yamnet']['scenarios'][scenario]['avg_time']
                    speedup = time1 / time2
                    print(f"  {scenario}:")
                    print(f"    Platform 1: {time1:.3f}s ({1/time1:.1f} FPS)")
                    print(f"    Platform 2: {time2:.3f}s ({1/time2:.1f} FPS)")
                    print(f"    Speedup: {speedup:.2f}x")
        
        # Compare system resources
        if 'resources' in results1 and 'resources' in results2:
            print("\n💻 Resource Usage Comparison:")
            print(f"  CPU Usage:")
            print(f"    Platform 1: {results1['resources']['cpu_avg']:.1f}% avg")
            print(f"    Platform 2: {results2['resources']['cpu_avg']:.1f}% avg")
            print(f"  Memory Usage:")
            print(f"    Platform 1: {results1['resources']['memory_avg']:.1f}% avg")
            print(f"    Platform 2: {results2['resources']['memory_avg']:.1f}% avg")
        
    except Exception as e:
        print(f"❌ Comparison failed: {e}")

def main():
    """Run comprehensive audio performance tests."""
    print("🎵 Audio Performance Test Suite")
    print("=" * 60)
    print("This will test audio processing performance for comparison")
    print("between Windows PC and Raspberry Pi platforms.")
    print()
    
    # Get system information
    system_info = get_system_info()
    platform_name = f"{system_info['platform']}_{system_info['architecture']}"
    
    print(f"🖥️  Platform: {system_info['platform']} {system_info['platform_release']}")
    print(f"🏗️  Architecture: {system_info['architecture']}")
    print(f"🧠 CPU: {system_info['processor']} ({system_info['cpu_count']} cores)")
    print(f"💾 Memory: {system_info['memory_total'] / (1024**3):.1f} GB")
    print(f"🐍 Python: {system_info['python_version']}")
    print()
    
    # Determine config file
    config_file = "config_raspberry_pi.env" if "raspberry" in platform_name.lower() or "arm" in platform_name.lower() else "config.env"
    print(f"📋 Using config: {config_file}")
    print()
    
    # Run tests
    results = {
        'timestamp': datetime.now().isoformat(),
        'system_info': system_info,
        'platform_name': platform_name,
        'config_file': config_file
    }
    
    # Test 1: YAMNet Performance
    yamnet_results = test_yamnet_performance(config_file, num_tests=20)
    if yamnet_results:
        results['yamnet'] = yamnet_results
    
    # Test 2: Audio Capture Performance
    audio_results = test_audio_capture_performance(config_file, duration=15)
    if audio_results:
        results['audio_capture'] = audio_results
    
    # Test 3: System Resources
    resource_results = test_system_resources(duration=20)
    if resource_results:
        results['resources'] = resource_results
    
    # Test 4: Integrated Performance
    integrated_results = test_integrated_performance(config_file, duration=30)
    if integrated_results:
        results['integrated'] = integrated_results
    
    # Save results
    results_file = save_results(results, platform_name)
    
    print(f"\n🎯 Performance Test Complete!")
    print(f"📊 Results saved to: {results_file}")
    print(f"\nTo compare with another platform:")
    print(f"  python test_audio_performance.py --compare {results_file} other_platform_results.json")
    
    return results_file

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Audio Performance Testing')
    parser.add_argument('--compare', nargs=2, help='Compare two result files')
    
    args = parser.parse_args()
    
    if args.compare:
        compare_results(args.compare[0], args.compare[1])
    else:
        main()
