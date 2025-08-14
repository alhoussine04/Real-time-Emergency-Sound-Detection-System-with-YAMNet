#!/usr/bin/env python3
"""
Inspect the YAMNet TF-Lite model file.
"""

import zipfile
import tensorflow as tf
import os

def inspect_model():
    """Inspect the model file."""
    model_path = "1.tflite"
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return False
    
    print(f"📁 Model file: {model_path}")
    print(f"📏 File size: {os.path.getsize(model_path) / 1024 / 1024:.2f} MB")
    
    # Try to load with TensorFlow Lite
    try:
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        
        # Get input and output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print("\n📥 Input details:")
        for i, detail in enumerate(input_details):
            print(f"  {i}: {detail['name']} - Shape: {detail['shape']} - Type: {detail['dtype']}")
        
        print("\n📤 Output details:")
        for i, detail in enumerate(output_details):
            print(f"  {i}: {detail['name']} - Shape: {detail['shape']} - Type: {detail['dtype']}")
        
        print("\n✅ Model loaded successfully with TensorFlow Lite")
        
    except Exception as e:
        print(f"❌ Failed to load model with TensorFlow Lite: {e}")
        return False
    
    # Try to check if it's a zip file (some TF-Lite models contain metadata)
    try:
        with zipfile.ZipFile(model_path, 'r') as zip_file:
            files = zip_file.namelist()
            print(f"\n📦 Model contains {len(files)} files:")
            for file in files:
                print(f"  - {file}")
            
            # Look for label file
            if 'yamnet_label_list.txt' in files:
                print("\n🏷️ Found label file!")
                with zip_file.open('yamnet_label_list.txt') as label_file:
                    labels = [line.decode('utf-8').strip() for line in label_file.readlines()]
                    print(f"   Labels: {len(labels)} classes")
                    print(f"   First 5: {labels[:5]}")
                    print(f"   Last 5: {labels[-5:]}")
            else:
                print("\n⚠️ No label file found in model")
                
    except zipfile.BadZipFile:
        print("\n📄 Model is not a zip file (no embedded metadata)")
    except Exception as e:
        print(f"\n❌ Error inspecting model as zip: {e}")
    
    return True

if __name__ == "__main__":
    print("YAMNet Model Inspector")
    print("=" * 30)
    inspect_model()
