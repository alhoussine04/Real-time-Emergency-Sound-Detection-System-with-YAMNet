# Test Audio Directory

This directory contains audio samples for performance evaluation testing.

## File Organization

Add your WAV files to this directory following these naming conventions:

### Target Sounds (Should be detected)

**Baby Crying Sounds:**
- baby_cry_1.wav
- baby_cry_2.wav  
- infant_crying.wav
- baby_cry_noisy.wav (optional - with background noise)

**Glass Breaking Sounds:**
- glass_break_1.wav
- window_shatter.wav
- bottle_break.wav
- glass_break_noisy.wav (optional - with background noise)

**Fire Alarm Sounds:**
- fire_alarm_1.wav
- smoke_alarm.wav
- emergency_beep.wav
- fire_alarm_noisy.wav (optional - with background noise)

### Background Sounds (Should NOT be detected)

**Background Noise:**
- tv_audio.wav
- conversation.wav
- music.wav
- white_noise.wav
- pink_noise.wav

**Negative Samples:**
- silence.wav

## Audio File Requirements

- **Format**: WAV files
- **Sample Rate**: 16kHz (recommended)
- **Channels**: Mono (recommended)
- **Duration**: 1-5 seconds per sample
- **Bit Depth**: 16-bit (recommended)

## Usage

1. Add your audio files to this directory following the naming conventions above
2. Run performance evaluation: `python run_performance_test.py`
3. The system will automatically detect and test all available WAV files

## Notes

- You don't need to have all the files listed above
- The system will test whatever WAV files are available
- Files not matching the expected names will be treated as background noise
- For consistent results, use similar audio characteristics (sample rate, duration, etc.)

## Alternative: Generate Synthetic Samples

If you don't have real audio samples, you can generate synthetic ones:
```bash
python run_performance_test.py --generate-audio
```

This will create all the files listed above with synthetic audio content.
