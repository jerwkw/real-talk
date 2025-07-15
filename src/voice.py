from audio import AudioCapture, AudioPlayback
from models import VoiceConverter, RVCConverter, BitNetConverter, TraditionalConverter
import numpy as np
import pyaudio
import threading
import time
from typing import Dict, Any

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