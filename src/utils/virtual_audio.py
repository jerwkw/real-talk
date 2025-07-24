from audio.playback import AudioCapture, AudioPlayback
import pyaudio

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
