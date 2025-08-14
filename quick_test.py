#!/usr/bin/env python3
"""
Quick test to verify the YAMNet classifier works.
"""

import sys
import os
import numpy as np

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_basic_imports():
    """Test basic imports."""
    print("Testing basic imports...")
    
    try:
        # Test TensorFlow Lite Runtime first (for Raspberry Pi), then others
        tf_available = False

        try:
            import tflite_runtime.interpreter as tflite
            print("✓ TensorFlow Lite Runtime imported successfully")
            tf_available = True
        except ImportError:
            try:
                from ai_edge_litert.interpreter import Interpreter
                print("✓ AI Edge LiteRT imported successfully")
                tf_available = True
            except ImportError:
                try:
                    import tensorflow as tf
                    print(f"✓ TensorFlow {tf.__version__} imported successfully")
                    tf_available = True
                except ImportError:
                    print("✗ No TensorFlow/TFLite runtime found")
                    print("  Please install one of:")
                    print("  - pip install tflite-runtime (Raspberry Pi)")
                    print("  - pip install tensorflow (PC)")
                    return False

        import numpy as np
        print(f"✓ NumPy {np.__version__} imported successfully")

        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False

def test_yamnet_model():
    """Test YAMNet model loading."""
    print("\nTesting YAMNet model...")
    
    try:
        from src.yamnet_classifier import YAMNetClassifier
        
        # Check if model file exists
        if not os.path.exists("1.tflite"):
            print("✗ Model file '1.tflite' not found")
            return False
        
        # Initialize classifier
        classifier = YAMNetClassifier("1.tflite")
        print("✓ YAMNet classifier initialized")
        
        # Test with dummy audio
        dummy_audio = np.zeros(15600, dtype=np.float32)
        classes, scores = classifier.classify(dummy_audio)
        
        if classes and scores:
            print(f"✓ Classification successful - Top: {classes[0]} ({scores[0]:.3f})")
            return True
        else:
            print("✗ Classification returned no results")
            return False
            
    except Exception as e:
        print(f"✗ YAMNet test failed: {e}")
        return False

def main():
    """Run quick tests."""
    print("Quick Test - Audio Monitoring System")
    print("=" * 40)
    
    # Test imports
    if not test_basic_imports():
        print("\n❌ Basic imports failed")
        return False
    
    # Test YAMNet
    if not test_yamnet_model():
        print("\n❌ YAMNet model test failed")
        return False
    
    print("\n✅ All tests passed! System is ready.")
    print("\nNext steps:")
    print("1. Configure Telegram bot in config.env")
    print("2. Run: python main.py --test-telegram")
    print("3. Run: python main.py")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        sys.exit(1)
