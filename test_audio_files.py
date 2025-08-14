#!/usr/bin/env python3
"""
Audio Performance Test using real WAV files from test_audio directory
Compare Windows PC vs Raspberry Pi performance with actual audio samples
"""

import sys
import os
import time
import platform
import numpy as np
import json
from datetime import datetime
import glob

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def load_audio_file(file_path):
    """Load audio file and convert to YAMNet format."""
    try:
        # Use basic wave module (more reliable)
        import wave
        with wave.open(file_path, 'rb') as wav_file:
            frames = wav_file.readframes(-1)
            sample_rate = wav_file.getframerate()
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()

            # Convert based on sample width
            if sample_width == 1:
                audio_data = np.frombuffer(frames, dtype=np.uint8).astype(np.float32) / 128.0 - 1.0
            elif sample_width == 2:
                audio_data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
            elif sample_width == 4:
                audio_data = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
            else:
                audio_data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0

        # Convert to mono if stereo
        if channels > 1:
            audio_data = audio_data.reshape(-1, channels)
            audio_data = np.mean(audio_data, axis=1)
        
        # Resample to 16kHz if needed (simple decimation/interpolation)
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
            # Take middle section
            start = (len(audio_data) - target_length) // 2
            audio_data = audio_data[start:start + target_length]
        elif len(audio_data) < target_length:
            # Pad with zeros
            padding = target_length - len(audio_data)
            audio_data = np.pad(audio_data, (0, padding), mode='constant')
        
        return audio_data.astype(np.float32)
        
    except Exception as e:
        print(f"❌ Failed to load {file_path}: {e}")
        return None

def categorize_audio_files():
    """Categorize audio files by expected detection type."""
    test_audio_dir = "test_audio"
    
    categories = {
        'baby_crying': [
            'baby_cry_1.wav', 'baby_cry_2.wav', 'infant_crying.wav'
        ],
        'glass_breaking': [
            'glass_break_1.wav', 'window_shatter.wav', 'bottle_break.wav'
        ],
        'fire_alarms': [
            'fire_alarm_1.wav', 'smoke_alarm.wav', 'emergency_beep.wav'
        ],
        'background_noise': [
            'tv_audio.wav', 'conversation.wav', 'music.wav', 
            'white_noise.wav', 'pink_noise.wav', 'silence.wav'
        ]
    }
    
    available_files = {}
    
    for category, filenames in categories.items():
        available_files[category] = []
        for filename in filenames:
            file_path = os.path.join(test_audio_dir, filename)
            if os.path.exists(file_path):
                available_files[category].append(file_path)
    
    return available_files

def test_audio_classification_performance(config_file="config.env"):
    """Test classification performance using real audio files."""
    print("🎵 Testing Audio Classification Performance")
    print("=" * 50)
    
    try:
        from src.yamnet_classifier import YAMNetClassifier
        
        # Initialize classifier
        print("Loading YAMNet model...")
        start_time = time.time()
        classifier = YAMNetClassifier("1.tflite")
        load_time = time.time() - start_time
        print(f"✓ Model loaded in {load_time:.2f}s")
        
        # Get available audio files
        audio_files = categorize_audio_files()
        
        results = {
            'load_time': load_time,
            'categories': {},
            'total_files': 0,
            'total_time': 0,
            'detections': []
        }
        
        target_sounds = ['Baby cry, infant cry', 'Glass', 'Fire alarm', 'Smoke detector, smoke alarm']
        
        for category, file_paths in audio_files.items():
            if not file_paths:
                continue
                
            print(f"\n📂 Testing {category} ({len(file_paths)} files):")
            
            category_results = {
                'files_tested': 0,
                'total_time': 0,
                'detections': [],
                'classification_times': []
            }
            
            for file_path in file_paths:
                filename = os.path.basename(file_path)
                print(f"  🎵 {filename}...")
                
                # Load audio
                audio_data = load_audio_file(file_path)
                if audio_data is None:
                    continue
                
                # Classify
                start = time.time()
                classes, scores = classifier.classify(audio_data)
                end = time.time()
                
                classification_time = end - start
                category_results['classification_times'].append(classification_time)
                category_results['total_time'] += classification_time
                category_results['files_tested'] += 1
                
                # Check for target sound detections
                detected_targets = []
                if classes and scores:
                    for i, (class_name, score) in enumerate(zip(classes[:5], scores[:5])):
                        for target in target_sounds:
                            if target.lower() in class_name.lower() or class_name.lower() in target.lower():
                                detected_targets.append({
                                    'target': target,
                                    'class': class_name,
                                    'confidence': score,
                                    'rank': i + 1
                                })
                
                # Log results
                if detected_targets:
                    best_detection = max(detected_targets, key=lambda x: x['confidence'])
                    print(f"    ✓ DETECTED: {best_detection['class']} ({best_detection['confidence']:.3f})")
                    category_results['detections'].append({
                        'file': filename,
                        'detection': best_detection,
                        'all_detections': detected_targets,
                        'top_prediction': {'class': classes[0], 'confidence': scores[0]} if classes else None
                    })
                else:
                    top_pred = f"{classes[0]} ({scores[0]:.3f})" if classes else "No prediction"
                    print(f"    - No target detected. Top: {top_pred}")
                
                print(f"    ⏱️  Classification time: {classification_time:.3f}s")
            
            # Category summary
            if category_results['files_tested'] > 0:
                avg_time = category_results['total_time'] / category_results['files_tested']
                detection_rate = len(category_results['detections']) / category_results['files_tested']
                
                print(f"  📊 Category Summary:")
                print(f"    Files tested: {category_results['files_tested']}")
                print(f"    Avg time: {avg_time:.3f}s ({1/avg_time:.1f} FPS)")
                print(f"    Detection rate: {detection_rate:.1%}")
                
                results['categories'][category] = category_results
                results['total_files'] += category_results['files_tested']
                results['total_time'] += category_results['total_time']
        
        # Overall summary
        if results['total_files'] > 0:
            overall_avg = results['total_time'] / results['total_files']
            print(f"\n🎯 Overall Performance:")
            print(f"  Total files: {results['total_files']}")
            print(f"  Total time: {results['total_time']:.2f}s")
            print(f"  Average per file: {overall_avg:.3f}s ({1/overall_avg:.1f} FPS)")
            
            results['overall_avg_time'] = overall_avg
            results['overall_fps'] = 1 / overall_avg
        
        return results
        
    except Exception as e:
        print(f"❌ Audio classification test failed: {e}")
        return None

def test_detection_accuracy():
    """Test detection accuracy for target vs background sounds."""
    print("\n🎯 Testing Detection Accuracy")
    print("=" * 40)
    
    try:
        from src.yamnet_classifier import YAMNetClassifier
        from dotenv import load_dotenv
        
        # Load config to get confidence threshold
        config_file = "config_raspberry_pi.env" if "arm" in platform.machine().lower() else "config.env"
        load_dotenv(config_file)
        confidence_threshold = float(os.getenv('CONFIDENCE_THRESHOLD', '0.3'))
        
        classifier = YAMNetClassifier("1.tflite")
        audio_files = categorize_audio_files()
        
        target_sounds = ['Baby cry, infant cry', 'Glass', 'Fire alarm', 'Smoke detector, smoke alarm']
        
        # Test target sounds (should be detected)
        target_categories = ['baby_crying', 'glass_breaking', 'fire_alarms']
        background_categories = ['background_noise']
        
        true_positives = 0
        false_negatives = 0
        true_negatives = 0
        false_positives = 0
        
        print(f"Using confidence threshold: {confidence_threshold}")
        
        # Test target sounds
        for category in target_categories:
            files = audio_files.get(category, [])
            print(f"\n📂 Testing {category} (should detect):")
            
            for file_path in files:
                filename = os.path.basename(file_path)
                audio_data = load_audio_file(file_path)
                if audio_data is None:
                    continue
                
                classes, scores = classifier.classify(audio_data)
                detected = False
                
                if classes and scores:
                    for class_name, score in zip(classes[:5], scores[:5]):
                        if score >= confidence_threshold:
                            for target in target_sounds:
                                if target.lower() in class_name.lower() or class_name.lower() in target.lower():
                                    detected = True
                                    break
                        if detected:
                            break
                
                if detected:
                    true_positives += 1
                    print(f"  ✓ {filename}: DETECTED (correct)")
                else:
                    false_negatives += 1
                    print(f"  ❌ {filename}: NOT detected (missed)")
        
        # Test background sounds
        for category in background_categories:
            files = audio_files.get(category, [])
            print(f"\n📂 Testing {category} (should NOT detect):")
            
            for file_path in files:
                filename = os.path.basename(file_path)
                audio_data = load_audio_file(file_path)
                if audio_data is None:
                    continue
                
                classes, scores = classifier.classify(audio_data)
                detected = False
                
                if classes and scores:
                    for class_name, score in zip(classes[:5], scores[:5]):
                        if score >= confidence_threshold:
                            for target in target_sounds:
                                if target.lower() in class_name.lower() or class_name.lower() in target.lower():
                                    detected = True
                                    break
                        if detected:
                            break
                
                if not detected:
                    true_negatives += 1
                    print(f"  ✓ {filename}: NOT detected (correct)")
                else:
                    false_positives += 1
                    print(f"  ❌ {filename}: DETECTED (false positive)")
        
        # Calculate metrics
        total = true_positives + false_negatives + true_negatives + false_positives
        
        if total > 0:
            accuracy = (true_positives + true_negatives) / total
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            print(f"\n📊 Detection Accuracy Results:")
            print(f"  True Positives: {true_positives}")
            print(f"  False Negatives: {false_negatives}")
            print(f"  True Negatives: {true_negatives}")
            print(f"  False Positives: {false_positives}")
            print(f"  Accuracy: {accuracy:.1%}")
            print(f"  Precision: {precision:.1%}")
            print(f"  Recall: {recall:.1%}")
            print(f"  F1 Score: {f1_score:.3f}")
            
            return {
                'true_positives': true_positives,
                'false_negatives': false_negatives,
                'true_negatives': true_negatives,
                'false_positives': false_positives,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score,
                'confidence_threshold': confidence_threshold
            }
        
        return None
        
    except Exception as e:
        print(f"❌ Accuracy test failed: {e}")
        return None

def main():
    """Run comprehensive audio file performance test."""
    print("🎵 Audio Performance Test - Using Real WAV Files")
    print("=" * 60)
    
    # System info
    system = platform.system()
    arch = platform.machine()
    print(f"🖥️  Platform: {system} {arch}")
    
    # Check available files
    audio_files = categorize_audio_files()
    total_files = sum(len(files) for files in audio_files.values())
    
    print(f"📁 Available test files: {total_files}")
    for category, files in audio_files.items():
        if files:
            print(f"  {category}: {len(files)} files")
    
    if total_files == 0:
        print("❌ No audio files found in test_audio directory!")
        print("Please add WAV files to test_audio/ directory")
        return
    
    # Determine config
    config_file = "config_raspberry_pi.env" if "arm" in arch.lower() else "config.env"
    print(f"📋 Using config: {config_file}")
    
    # Run tests
    results = {
        'timestamp': datetime.now().isoformat(),
        'platform': f"{system} {arch}",
        'config_file': config_file,
        'total_files': total_files
    }
    
    # Test 1: Classification Performance
    perf_results = test_audio_classification_performance(config_file)
    if perf_results:
        results['performance'] = perf_results
    
    # Test 2: Detection Accuracy
    accuracy_results = test_detection_accuracy()
    if accuracy_results:
        results['accuracy'] = accuracy_results
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    platform_name = f"{system}_{arch}".replace(" ", "_")
    filename = f"audio_test_results_{platform_name}_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to: {filename}")
    
    # Summary
    print(f"\n🎯 Test Summary:")
    if perf_results:
        print(f"  Performance: {perf_results.get('overall_fps', 0):.1f} FPS average")
    if accuracy_results:
        print(f"  Accuracy: {accuracy_results.get('accuracy', 0):.1%}")
        print(f"  F1 Score: {accuracy_results.get('f1_score', 0):.3f}")
    
    print(f"\nTo compare with another platform:")
    print(f"  python test_audio_files.py --compare {filename} other_results.json")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Audio Performance Test with WAV files')
    parser.add_argument('--compare', nargs=2, help='Compare two result files')
    
    args = parser.parse_args()
    
    if args.compare:
        # Simple comparison
        try:
            with open(args.compare[0], 'r') as f:
                results1 = json.load(f)
            with open(args.compare[1], 'r') as f:
                results2 = json.load(f)
            
            print(f"🔍 Comparison: {args.compare[0]} vs {args.compare[1]}")
            print("=" * 60)
            
            if 'performance' in results1 and 'performance' in results2:
                fps1 = results1['performance'].get('overall_fps', 0)
                fps2 = results2['performance'].get('overall_fps', 0)
                print(f"Performance: {fps1:.1f} FPS vs {fps2:.1f} FPS")
                if fps2 > 0:
                    print(f"Speedup: {fps1/fps2:.2f}x")
            
            if 'accuracy' in results1 and 'accuracy' in results2:
                acc1 = results1['accuracy'].get('accuracy', 0)
                acc2 = results2['accuracy'].get('accuracy', 0)
                print(f"Accuracy: {acc1:.1%} vs {acc2:.1%}")
                
        except Exception as e:
            print(f"❌ Comparison failed: {e}")
    else:
        main()
