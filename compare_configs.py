#!/usr/bin/env python3
"""
Configuration Comparison Tool - Compare Windows PC vs Raspberry Pi settings
"""

import os
from dotenv import load_dotenv

def load_config(config_file):
    """Load configuration from file."""
    if not os.path.exists(config_file):
        return None
    
    load_dotenv(config_file)
    
    return {
        'telegram_bot_token': os.getenv('TELEGRAM_BOT_TOKEN', ''),
        'telegram_chat_id': os.getenv('TELEGRAM_CHAT_ID', ''),
        'sample_rate': int(os.getenv('SAMPLE_RATE', '16000')),
        'chunk_size': int(os.getenv('CHUNK_SIZE', '1024')),
        'channels': int(os.getenv('CHANNELS', '1')),
        'audio_device_index': int(os.getenv('AUDIO_DEVICE_INDEX', '-1')),
        'confidence_threshold': float(os.getenv('CONFIDENCE_THRESHOLD', '0.3')),
        'detection_cooldown': float(os.getenv('DETECTION_COOLDOWN', '5.0')),
        'buffer_duration': float(os.getenv('BUFFER_DURATION', '0.975')),
        'target_sounds': os.getenv('TARGET_SOUNDS', '').split(';'),
        'log_level': os.getenv('LOG_LEVEL', 'INFO'),
        'log_file': os.getenv('LOG_FILE', ''),
    }

def compare_configs():
    """Compare Windows PC and Raspberry Pi configurations."""
    print("⚙️  Configuration Comparison: Windows PC vs Raspberry Pi")
    print("=" * 65)
    
    # Load configurations
    pc_config = load_config('config.env')
    pi_config = load_config('config_raspberry_pi.env')
    
    if not pc_config:
        print("❌ config.env not found")
        return
    
    if not pi_config:
        print("❌ config_raspberry_pi.env not found")
        return
    
    # Compare settings
    settings = [
        ('Sample Rate', 'sample_rate', 'Hz'),
        ('Chunk Size', 'chunk_size', 'samples'),
        ('Channels', 'channels', ''),
        ('Confidence Threshold', 'confidence_threshold', ''),
        ('Detection Cooldown', 'detection_cooldown', 's'),
        ('Buffer Duration', 'buffer_duration', 's'),
        ('Log Level', 'log_level', ''),
    ]
    
    print(f"{'Setting':<20} {'Windows PC':<15} {'Raspberry Pi':<15} {'Difference':<15}")
    print("-" * 65)
    
    for name, key, unit in settings:
        pc_val = pc_config.get(key, 'N/A')
        pi_val = pi_config.get(key, 'N/A')
        
        # Format values
        if isinstance(pc_val, float):
            pc_str = f"{pc_val:.3f}{unit}"
            pi_str = f"{pi_val:.3f}{unit}"
        else:
            pc_str = f"{pc_val}{unit}"
            pi_str = f"{pi_val}{unit}"
        
        # Calculate difference
        if isinstance(pc_val, (int, float)) and isinstance(pi_val, (int, float)):
            if pc_val != 0:
                diff_pct = ((pi_val - pc_val) / pc_val) * 100
                diff_str = f"{diff_pct:+.1f}%"
            else:
                diff_str = "N/A"
        else:
            diff_str = "Same" if pc_val == pi_val else "Different"
        
        print(f"{name:<20} {pc_str:<15} {pi_str:<15} {diff_str:<15}")
    
    # Performance implications
    print(f"\n🚀 Performance Implications:")
    print("-" * 40)
    
    # Chunk size comparison
    pc_chunk = pc_config['chunk_size']
    pi_chunk = pi_config['chunk_size']
    
    if pi_chunk > pc_chunk:
        ratio = pi_chunk / pc_chunk
        print(f"✓ Raspberry Pi uses {ratio:.1f}x larger chunks")
        print(f"  → Reduces CPU load by ~{(1-1/ratio)*100:.0f}%")
        print(f"  → Increases latency by ~{(ratio-1)*pc_chunk/16000*1000:.0f}ms")
    
    # Confidence threshold comparison
    pc_conf = pc_config['confidence_threshold']
    pi_conf = pi_config['confidence_threshold']
    
    if pi_conf != pc_conf:
        if pi_conf > pc_conf:
            print(f"✓ Raspberry Pi uses higher confidence threshold")
            print(f"  → Fewer false positives")
            print(f"  → May miss some weak signals")
        else:
            print(f"✓ Raspberry Pi uses lower confidence threshold")
            print(f"  → More sensitive detection")
            print(f"  → May have more false positives")
    
    # Cooldown comparison
    pc_cool = pc_config['detection_cooldown']
    pi_cool = pi_config['detection_cooldown']
    
    if pi_cool > pc_cool:
        ratio = pi_cool / pc_cool
        print(f"✓ Raspberry Pi uses {ratio:.1f}x longer cooldown")
        print(f"  → Reduces notification spam")
        print(f"  → Reduces processing load")
    
    # Target sounds comparison
    pc_sounds = set(s.strip() for s in pc_config['target_sounds'] if s.strip())
    pi_sounds = set(s.strip() for s in pi_config['target_sounds'] if s.strip())
    
    print(f"\n🎯 Target Sounds:")
    print(f"  Common sounds: {len(pc_sounds & pi_sounds)}")
    print(f"  PC only: {len(pc_sounds - pi_sounds)}")
    print(f"  Pi only: {len(pi_sounds - pc_sounds)}")
    
    if pc_sounds != pi_sounds:
        print(f"  Different target sounds detected!")

def recommend_settings():
    """Provide platform-specific recommendations."""
    print(f"\n💡 Platform-Specific Recommendations:")
    print("=" * 50)
    
    print(f"\n🖥️  Windows PC (High Performance):")
    print(f"  CHUNK_SIZE=1024          # Real-time processing")
    print(f"  CONFIDENCE_THRESHOLD=0.2  # Sensitive detection")
    print(f"  DETECTION_COOLDOWN=5.0    # Quick notifications")
    print(f"  LOG_LEVEL=INFO           # Detailed logging")
    
    print(f"\n🍓 Raspberry Pi 4 (Balanced):")
    print(f"  CHUNK_SIZE=2048          # Reduced CPU load")
    print(f"  CONFIDENCE_THRESHOLD=0.25 # Balanced sensitivity")
    print(f"  DETECTION_COOLDOWN=10.0   # Reduced processing")
    print(f"  LOG_LEVEL=INFO           # Standard logging")
    
    print(f"\n🍓 Raspberry Pi 3B+ (Conservative):")
    print(f"  CHUNK_SIZE=4096          # Minimize CPU load")
    print(f"  CONFIDENCE_THRESHOLD=0.3  # Reduce false positives")
    print(f"  DETECTION_COOLDOWN=15.0   # Minimal processing")
    print(f"  LOG_LEVEL=WARNING        # Reduced logging")
    
    print(f"\n⚡ Performance Tuning Tips:")
    print(f"  • Larger CHUNK_SIZE = Lower CPU, Higher latency")
    print(f"  • Higher CONFIDENCE_THRESHOLD = Fewer false positives")
    print(f"  • Longer DETECTION_COOLDOWN = Less processing load")
    print(f"  • Fewer TARGET_SOUNDS = Faster processing")

def main():
    """Run configuration comparison."""
    compare_configs()
    recommend_settings()
    
    print(f"\n🔧 To test performance with current settings:")
    print(f"  python test_audio_quick.py")
    print(f"\n📊 For detailed performance analysis:")
    print(f"  python test_audio_performance.py")

if __name__ == "__main__":
    main()
