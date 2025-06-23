# Voice Conversion Testing Framework

import numpy as np
import pyaudio
import threading
import queue
import time
from abc import ABC, abstractmethod
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

# VOICE CONVERSION MODEL INTERFACE

class VoiceConverter(ABC):
    """Abstract base class for all voice conversion models"""
    
    def __init__(self, name: str):
        self.name = name
        self.is_loaded = False
        self.latency_ms = 0
        self.model_info = {}
        
    @abstractmethod
    def load_model(self, model_path: str, **kwargs):
        """Load the voice conversion model"""
        pass
        
    @abstractmethod
    def convert(self, audio_input: np.ndarray) -> np.ndarray:
        """Convert audio using the loaded model"""
        pass
        
    @abstractmethod
    def unload_model(self):
        """Unload model and free resources"""
        pass
        
    def get_latency(self) -> float:
        """Return model latency in milliseconds"""
        return self.latency_ms
        
    def get_model_info(self) -> Dict[str, Any]:
        """Return model information"""
        return {
            "name": self.name,
            "loaded": self.is_loaded,
            "latency_ms": self.latency_ms,
            **self.model_info
        }

# MODEL IMPLEMENTATIONS ( in progess )

class RVCConverter(VoiceConverter):
    """RVC (Retrieval-based Voice Conversion) implementation"""
    
    def __init__(self):
        super().__init__("RVC")
        self.model = None
        
    def load_model(self, model_path: str, **kwargs):
        """Load RVC model"""
        try:
            # Placeholder for actual RVC model loading
            # In reality, you'd import and initialize the RVC library
            print(f"Loading RVC model from {model_path}")
            self.model = "rvc_model_placeholder"  # Replace with actual model
            self.is_loaded = True
            self.model_info = {
                "type": "RVC",
                "model_path": model_path,
                "sample_rate": kwargs.get("sample_rate", 44100)
            }
            return True
        except Exception as e:
            print(f"Failed to load RVC model: {e}")
            return False
            
    def convert(self, audio_input: np.ndarray) -> np.ndarray:
        """Convert audio using RVC"""
        if not self.is_loaded:
            return audio_input
            
        start_time = time.time()
        
        # Placeholder for actual RVC conversion
        # In reality, you'd call the RVC inference function
        converted_audio = audio_input  # Replace with actual conversion
        
        # Calculate latency
        self.latency_ms = (time.time() - start_time) * 1000
        
        return converted_audio
        
    def unload_model(self):
        """Unload RVC model"""
        self.model = None
        self.is_loaded = False

class BitNetConverter(VoiceConverter):
    """BitNet-based voice conversion implementation"""
    
    def __init__(self):
        super().__init__("BitNet")
        self.model = None
        
    def load_model(self, model_path: str, **kwargs):
        """Load BitNet model"""
        try:
            print(f"Loading BitNet model from {model_path}")
            # Placeholder for BitNet model loading
            self.model = "bitnet_model_placeholder"
            self.is_loaded = True
            self.model_info = {
                "type": "BitNet",
                "model_path": model_path,
                "quantization": "1-bit",
                "sample_rate": kwargs.get("sample_rate", 44100)
            }
            return True
        except Exception as e:
            print(f"Failed to load BitNet model: {e}")
            return False
            
    def convert(self, audio_input: np.ndarray) -> np.ndarray:
        """Convert audio using BitNet"""
        if not self.is_loaded:
            return audio_input
            
        start_time = time.time()
        
        # Placeholder for BitNet conversion
        # Your experimental BitNet implementation would go here
        converted_audio = audio_input * 0.8  # Placeholder transformation
        
        self.latency_ms = (time.time() - start_time) * 1000
        return converted_audio
        
    def unload_model(self):
        """Unload BitNet model"""
        self.model = None
        self.is_loaded = False

class TraditionalConverter(VoiceConverter):
    """Traditional DSP-based voice conversion"""
    
    def __init__(self):
        super().__init__("Traditional DSP")
        self.pitch_shift = 1.0
        self.formant_shift = 1.0
        
    def load_model(self, model_path: str, **kwargs):
        """Load traditional processing parameters"""
        self.pitch_shift = kwargs.get("pitch_shift", 1.0)
        self.formant_shift = kwargs.get("formant_shift", 1.0)
        self.is_loaded = True
        self.model_info = {
            "type": "Traditional DSP",
            "pitch_shift": self.pitch_shift,
            "formant_shift": self.formant_shift
        }
        return True
        
    def convert(self, audio_input: np.ndarray) -> np.ndarray:
        """Apply traditional voice conversion"""
        start_time = time.time()
        
        # Simple pitch shifting using resampling (placeholder)
        # In reality, you'd use proper PSOLA or phase vocoder
        if self.pitch_shift != 1.0:
            # Crude pitch shifting by resampling
            new_length = int(len(audio_input) / self.pitch_shift)
            converted_audio = np.interp(
                np.linspace(0, len(audio_input)-1, new_length),
                np.arange(len(audio_input)),
                audio_input
            ).astype(np.int16)
        else:
            converted_audio = audio_input
            
        self.latency_ms = (time.time() - start_time) * 1000
        return converted_audio
        
    def unload_model(self):
        """No model to unload for traditional methods"""
        self.is_loaded = False

# PERFORMANCE MONITORING

class PerformanceMonitor:
    """Monitor performance metrics for voice conversion"""
    
    def __init__(self):
        self.metrics = {
            "latency_samples": [],
            "cpu_usage": [],
            "memory_usage": [],
            "audio_quality": []
        }
        
    def record_latency(self, latency_ms: float):
        """Record latency measurement"""
        self.metrics["latency_samples"].append(latency_ms)
        
    def get_average_latency(self) -> float:
        """Get average latency"""
        if not self.metrics["latency_samples"]:
            return 0.0
        return np.mean(self.metrics["latency_samples"])
        
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        latency_samples = self.metrics["latency_samples"]
        return {
            "avg_latency_ms": np.mean(latency_samples) if latency_samples else 0,
            "max_latency_ms": np.max(latency_samples) if latency_samples else 0,
            "min_latency_ms": np.min(latency_samples) if latency_samples else 0,
            "total_samples": len(latency_samples)
        }
        
    def reset_metrics(self):
        """Reset all metrics"""
        for key in self.metrics:
            self.metrics[key].clear()

# MAIN FRAMEWORK CLASS

class VoiceConversionFramework:
    """Main framework for testing different voice conversion models"""
    
    def __init__(self, sample_rate=44100, chunk_size=1024):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        
        # Audio components
        self.audio_capture = AudioCapture(sample_rate, chunk_size)
        self.audio_playback = AudioPlayback(sample_rate, chunk_size)
        
        # Available converters
        self.converters = {
            "rvc": RVCConverter(),
            "bitnet": BitNetConverter(),
            "traditional": TraditionalConverter()
        }
        
        self.current_converter = None
        self.performance_monitor = PerformanceMonitor()
        self.is_running = False
        self.processing_thread = None
        
    def list_available_models(self):
        """List all available conversion models"""
        return list(self.converters.keys())
        
    def load_model(self, model_name: str, model_path: str = "", **kwargs):
        """Load a specific voice conversion model"""
        if model_name not in self.converters:
            print(f"Model {model_name} not available")
            return False
            
        converter = self.converters[model_name]
        success = converter.load_model(model_path, **kwargs)
        
        if success:
            self.current_converter = converter
            print(f"Successfully loaded {model_name} model")
        else:
            print(f"Failed to load {model_name} model")
            
        return success
        
    def start_real_time_conversion(self):
        """Start real-time voice conversion"""
        if not self.current_converter or not self.current_converter.is_loaded:
            print("No model loaded. Please load a model first.")
            return False
            
        self.is_running = True
        self.audio_capture.start_capture()
        self.audio_playback.start_playback()
        
        # Start processing thread
        self.processing_thread = threading.Thread(target=self._processing_loop)
        self.processing_thread.start()
        
        print(f"Started real-time conversion using {self.current_converter.name}")
        return True
        
    def _processing_loop(self):
        """Main processing loop for real-time conversion"""
        while self.is_running:
            # Get audio chunk from input
            audio_chunk = self.audio_capture.get_audio_chunk()
            
            if audio_chunk is not None:
                # Convert audio using current model
                start_time = time.time()
                converted_audio = self.current_converter.convert(audio_chunk)
                processing_time = (time.time() - start_time) * 1000
                
                # Record performance metrics
                self.performance_monitor.record_latency(processing_time)
                
                # Send to output
                self.audio_playback.play_audio(converted_audio)
            else:
                # Small sleep to prevent busy waiting
                time.sleep(0.001)
                
    def stop_conversion(self):
        """Stop real-time voice conversion"""
        self.is_running = False
        
        if self.processing_thread:
            self.processing_thread.join()
            
        self.audio_capture.stop_capture()
        self.audio_playback.stop_playback()
        
        print("Stopped voice conversion")
        
    def get_performance_stats(self):
        """Get current performance statistics"""
        stats = self.performance_monitor.get_stats()
        if self.current_converter:
            stats["current_model"] = self.current_converter.name
        return stats
        
    def switch_model(self, model_name: str, model_path: str = "", **kwargs):
        """Switch to a different model during runtime"""
        was_running = self.is_running
        
        if was_running:
            self.stop_conversion()
            
        # Unload current model
        if self.current_converter:
            self.current_converter.unload_model()
            
        # Load new model
        success = self.load_model(model_name, model_path, **kwargs)
        
        if success and was_running:
            self.start_real_time_conversion()
            
        return success

# VIRTUAL AUDIO DEVICE SYSTEM

class VirtualAudioDevice:
    """Creates a virtual audio device for applications to use"""
    
    def __init__(self, framework):
        self.framework = framework
        self.virtual_cable = None
        self.device_name = "Voice Changer Virtual Device"
        
    def setup_virtual_device(self):
        """Set up virtual audio routing"""
        print(f"Setting up virtual audio device: {self.device_name}")
        print("Applications can now select '{self.device_name}' as their microphone")
        
        # In a real implementation, you'd:
        # 1. Create a virtual audio driver (Windows: Virtual Audio Cable, VAC)
        # 2. Route processed audio to this virtual device
        # 3. Applications see this as a real microphone
        
    def list_audio_devices(self):
        """List all available audio input devices"""
        audio = pyaudio.PyAudio()
        devices = []
        
        print("\nAvailable Audio Input Devices:")
        for i in range(audio.get_device_count()):
            info = audio.get_device_info_by_index(i)
            if info['maxInputChannels'] > 0:
                devices.append({
                    'index': i,
                    'name': info['name'],
                    'channels': info['maxInputChannels'],
                    'sample_rate': int(info['defaultSampleRate'])
                })
                print(f"  {i}: {info['name']} (Channels: {info['maxInputChannels']})")
        
        audio.terminate()
        return devices
        
    def select_input_device(self, device_index):
        """Select specific input device for capture"""
        self.framework.audio_capture.input_device_index = device_index
        print(f"Selected input device index: {device_index}")

class GameVoiceChanger(VoiceConversionFramework):
    """Extended framework specifically for game voice changing"""
    
    def __init__(self, sample_rate=44100, chunk_size=512):  # Smaller chunks for lower latency
        super().__init__(sample_rate, chunk_size)
        self.virtual_device = VirtualAudioDevice(self)
        self.target_latency_ms = 50  # Target latency for gaming
        
    def setup_for_gaming(self):
        """Configure optimal settings for gaming use"""
        print("=== Gaming Voice Changer Setup ===")
        
        # List available devices
        devices = self.virtual_device.list_audio_devices()
        
        # Let user select input device
        print("\nSelect your microphone:")
        try:
            device_choice = int(input("Enter device number: "))
            self.virtual_device.select_input_device(device_choice)
        except (ValueError, IndexError):
            print("Using default input device")
            
        # Set up virtual output
        self.virtual_device.setup_virtual_device()
        
        print(f"\nSetup Instructions:")
        print(f"1. In your game/Discord/app, select '{self.virtual_device.device_name}' as microphone")
        print(f"2. Start voice conversion with your chosen model")
        print(f"3. Your voice will be processed and sent to the virtual device")
        print(f"4. Target latency: {self.target_latency_ms}ms")
        
    def optimize_for_low_latency(self):
        """Optimize settings for minimal latency"""
        # Use smaller buffer sizes
        self.chunk_size = 256  # ~5.8ms at 44.1kHz
        
        # Recreate audio objects with new settings
        self.audio_capture = AudioCapture(self.sample_rate, self.chunk_size)
        self.audio_playback = AudioPlayback(self.sample_rate, self.chunk_size)
        
        print(f"Optimized for low latency: {self.chunk_size} samples ({self.chunk_size/self.sample_rate*1000:.1f}ms)")

# =============================================================================
# 7. ENHANCED AUDIO CAPTURE WITH DEVICE SELECTION
# =============================================================================

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