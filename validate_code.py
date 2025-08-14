#!/usr/bin/env python3
"""
Code validation script to ensure all Python files compile without syntax errors.
"""

import sys
import os
import py_compile
from pathlib import Path

def validate_file(file_path):
    """Validate a single Python file."""
    try:
        py_compile.compile(file_path, doraise=True)
        print(f"[OK] {file_path}")
        return True
    except py_compile.PyCompileError as e:
        print(f"[FAIL] {file_path}: {e}")
        return False
    except Exception as e:
        print(f"[ERROR] {file_path}: {e}")
        return False

def main():
    """Validate all Python files in the project."""
    print("=" * 50)
    print("CODE VALIDATION")
    print("=" * 50)
    
    # List of Python files to validate
    python_files = [
        "main.py",
        "setup.py",
        "test_system.py",
        "validate_code.py",
        "src/audio_monitor.py",
        "src/yamnet_classifier.py",
        "src/audio_capture.py",
        "src/telegram_notifier.py",
        "src/sound_detector.py"
    ]
    
    passed = 0
    total = len(python_files)
    
    print(f"Validating {total} Python files...\n")
    
    for file_path in python_files:
        if os.path.exists(file_path):
            if validate_file(file_path):
                passed += 1
        else:
            print(f"[MISSING] {file_path}")
    
    print("\n" + "=" * 50)
    print(f"VALIDATION RESULTS: {passed}/{total} files passed")
    print("=" * 50)
    
    if passed == total:
        print("[SUCCESS] All Python files compile successfully!")
        return 0
    else:
        print("[ERROR] Some files have syntax errors.")
        return 1

if __name__ == '__main__':
    sys.exit(main())