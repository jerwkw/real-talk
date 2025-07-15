# Voice Conversion Testing Framework

import numpy as np
import pyaudio
import threading
import queue
import time
from typing import Optional, Dict, Any
import json
import wave

# AUDIO CAPTURE & PLAYBACK COMPONENTS

class AudioCapture:
    """Handles real-time audio input from microphone"""
    
    def __init__(self, sample_rate=44100, chunk_size=1024, channels=1):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.channels = channels
        self.format = pyaudio.paInt16
        
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.is_recording = False
        self.audio_queue = queue.Queue()
        
    def start_capture(self):
        """Start capturing audio from microphone"""
        self.stream = self.audio.open(
            format=self.format,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size,
            stream_callback=self._audio_callback
        )
        self.is_recording = True
        self.stream.start_stream()
        
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Callback function for audio stream"""
        if self.is_recording:
            # Convert bytes to numpy array
            audio_data = np.frombuffer(in_data, dtype=np.int16)
            self.audio_queue.put(audio_data)
        return (None, pyaudio.paContinue)
        
    def get_audio_chunk(self):
        """Get the next audio chunk from queue"""
        try:
            return self.audio_queue.get_nowait()
        except queue.Empty:
            return None
            
    def stop_capture(self):
        """Stop audio capture"""
        self.is_recording = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.audio.terminate()

class AudioPlayback:
    """Handles real-time audio output to speakers"""
    
    def __init__(self, sample_rate=44100, chunk_size=1024, channels=1):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.channels = channels
        self.format = pyaudio.paInt16
        
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.playback_queue = queue.Queue()
        self.is_playing = False
        
    def start_playback(self):
        """Start audio playback"""
        self.stream = self.audio.open(
            format=self.format,
            channels=self.channels,
            rate=self.sample_rate,
            output=True,
            frames_per_buffer=self.chunk_size,
            stream_callback=self._playback_callback
        )
        self.is_playing = True
        self.stream.start_stream()
        
    def _playback_callback(self, in_data, frame_count, time_info, status):
        """Callback for audio playback"""
        try:
            audio_data = self.playback_queue.get_nowait()
            return (audio_data.tobytes(), pyaudio.paContinue)
        except queue.Empty:
            # Return silence if no audio available
            silence = np.zeros(frame_count, dtype=np.int16)
            return (silence.tobytes(), pyaudio.paContinue)
            
    def play_audio(self, audio_data):
        """Add audio data to playback queue"""
        self.playback_queue.put(audio_data)
        
    def stop_playback(self):
        """Stop audio playback"""
        self.is_playing = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.audio.terminate()


class EnhancedAudioCapture(AudioCapture):
    """Audio capture with device selection capabilities"""
    
    def __init__(self, sample_rate=44100, chunk_size=1024, channels=1, input_device_index=None):
        super().__init__(sample_rate, chunk_size, channels)
        self.input_device_index = input_device_index
        
    def start_capture(self):
        """Start capturing audio from specified device"""
        self.stream = self.audio.open(
            format=self.format,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            input_device_index=self.input_device_index,  # Specify device
            frames_per_buffer=self.chunk_size,
            stream_callback=self._audio_callback
        )
        self.is_recording = True
        self.stream.start_stream()

# SYSTEM INTEGRATION UTILITIES

def install_virtual_audio_driver():
    """Instructions for installing virtual audio drivers"""
    print("""
=== Virtual Audio Driver Installation ===

For Windows:
1. Download VB-Audio Virtual Cable: https://vb-audio.com/Cable/
2. Install and restart your computer
3. You'll see "CABLE Input" as a playback device and "CABLE Output" as recording device

For macOS:
1. Download Soundflower: https://github.com/mattingalls/Soundflower
2. Or use BlackHole: https://github.com/ExistentialAudio/BlackHole

For Linux:
1. Use PulseAudio virtual devices:
   pacmd load-module module-null-sink sink_name=virtual_mic
   pacmd load-module module-loopback source=virtual_mic.monitor

After installation, this program will route processed audio to the virtual device.
""")

def check_audio_latency():
    """Test audio system latency"""
    print("Testing audio system latency...")
    
    # Simple latency test
    audio = pyaudio.PyAudio()
    try:
        # Test different buffer sizes
        for chunk_size in [128, 256, 512, 1024, 2048]:
            latency_ms = (chunk_size / 44100) * 1000
            print(f"Buffer size {chunk_size}: ~{latency_ms:.1f}ms latency")
    finally:
        audio.terminate()

# USAGE EXAMPLES FOR GAMING

def gaming_setup_wizard():
    """Interactive setup for gaming use"""
    print("=== Voice Changer Gaming Setup Wizard ===")
    
    # Check for virtual audio drivers
    print("\n1. Checking system requirements...")
    install_virtual_audio_driver()
    
    input("Press Enter after installing virtual audio driver...")
    
    # Initialize gaming voice changer
    game_changer = GameVoiceChanger(chunk_size=256)  # Low latency
    
    # Run setup
    game_changer.setup_for_gaming()
    
    # Load a model
    print("\nSelect voice conversion model:")
    print("1. Traditional (Fastest, ~5ms)")
    print("2. BitNet (Fast, ~20ms)")
    print("3. RVC (Best quality, ~100ms)")
    
    choice = input("Enter choice (1-3): ")
    
    if choice == "1":
        game_changer.load_model("traditional", pitch_shift=1.3)
    elif choice == "2":
        game_changer.load_model("bitnet", "path/to/bitnet/model.pth")
    elif choice == "3":
        game_changer.load_model("rvc", "path/to/rvc/model.pth")
    
    return game_changer

def main():
    """Main application entry point"""
    print("Voice Changer Application")
    print("1. Standard Mode (speakers output)")
    print("2. Gaming Mode (virtual device output)")
    
    mode = input("Select mode (1-2): ")
    
    if mode == "2":
        # Gaming mode
        changer = gaming_setup_wizard()
        
        try:
            changer.start_real_time_conversion()
            print("\nVoice changer active! Commands:")
            print("'stats' - Show performance stats")
            print("'latency' - Check current latency")
            print("'quit' - Exit")
            
            while True:
                cmd = input().strip().lower()
                if cmd == 'quit':
                    break
                elif cmd == 'stats':
                    stats = changer.get_performance_stats()
                    print(f"Average latency: {stats['avg_latency_ms']:.1f}ms")
                    if stats['avg_latency_ms'] > 50:
                        print("⚠️  High latency detected - consider switching to faster model")
                elif cmd == 'latency':
                    check_audio_latency()
                    
        finally:
            changer.stop_conversion()
    else:
        # Standard mode (original implementation)
        framework = VoiceConversionFramework()
        
        try:
            framework.load_model("traditional", pitch_shift=1.2)
            framework.start_real_time_conversion()
            
            print("Voice conversion started. Commands:")
            print("'traditional', 'bitnet', 'rvc' - Switch models")
            print("'stats' - Performance stats")
            print("'quit' - Exit")
            
            while True:
                cmd = input().strip().lower()
                if cmd == 'quit':
                    break
                elif cmd in ['traditional', 'bitnet', 'rvc']:
                    framework.switch_model(cmd)
                elif cmd == 'stats':
                    stats = framework.get_performance_stats()
                    print("Performance Stats:", json.dumps(stats, indent=2))
                    
        finally:
            framework.stop_conversion()

if __name__ == "__main__":
    main()