import os
import sys
import time
import argparse
import numpy as np
import pyaudio
from pathlib import Path

# Ensure parent directory is in path
sys.path.append(str(Path(__file__).resolve().parent))

from models.models import DistilledONNXConverter

def list_audio_devices():
    """List all available audio input and output devices"""
    audio = pyaudio.PyAudio()
    print("\n================== AVAILABLE AUDIO DEVICES ==================")
    print(f"{'Index':<6} | {'Device Name':<45} | {'Inputs':<6} | {'Outputs':<7} | {'Sample Rate':<11}")
    print("-" * 85)
    
    for i in range(audio.get_device_count()):
        try:
            info = audio.get_device_info_by_index(i)
            name = info.get('name', 'Unknown')
            max_in = info.get('maxInputChannels', 0)
            max_out = info.get('maxOutputChannels', 0)
            default_sr = int(info.get('defaultSampleRate', 44100))
            
            # Print clean table row
            print(f"{i:<6} | {name[:45]:<45} | {max_in:<6} | {max_out:<7} | {default_sr:<11} Hz")
        except Exception as e:
            continue
            
    print("=============================================================\n")
    audio.terminate()

def run_inference_loop(model_path: str, vocos_path: str, input_idx: int, output_idx: int, chunk_size: int = 512):
    """Real-time audio processing loop"""
    # 1. Load model converter
    converter = DistilledONNXConverter()
    print("Loading models...")
    success = converter.load_model(model_path, vocos_onnx_path=vocos_path)
    if not success:
        print("Error: Could not load the distilled ONNX model.")
        sys.exit(1)
        
    print("Models loaded successfully!")
    
    # 2. Setup audio streams
    audio = pyaudio.PyAudio()
    
    # We use 24kHz as it is standard for Vocos and distillation pipeline
    sample_rate = 24000
    channels = 1
    audio_format = pyaudio.paInt16
    
    # Verify selected device indexes
    print(f"Opening input device (Index {input_idx})...")
    print(f"Opening output device (Index {output_idx})...")
    
    try:
        input_stream = audio.open(
            format=audio_format,
            channels=channels,
            rate=sample_rate,
            input=True,
            input_device_index=input_idx if input_idx >= 0 else None,
            frames_per_buffer=chunk_size
        )
        
        output_stream = audio.open(
            format=audio_format,
            channels=channels,
            rate=sample_rate,
            output=True,
            output_device_index=output_idx if output_idx >= 0 else None,
            frames_per_buffer=chunk_size
        )
    except Exception as e:
        print(f"\nFailed to open audio streams directly at 24000Hz: {e}")
        print("Please check your device index numbers or ensure that your devices support 24000Hz sampling rate.")
        audio.terminate()
        sys.exit(1)
        
    print("\n=============================================================")
    print("Voice Conversion is ACTIVE!")
    print("Instructions:")
    print("1. Speak into your microphone.")
    print("2. Set your game / Discord microphone to the virtual cable device.")
    print("3. Press CTRL+C in this terminal window to stop conversion.")
    print("=============================================================\n")
    
    latencies = []
    
    try:
        while True:
            # Read input chunk
            # exception_on_overflow=False prevents crashes when gaming frames drop
            data_bytes = input_stream.read(chunk_size, exception_on_overflow=False)
            
            # Convert bytes to numpy array
            audio_in = np.frombuffer(data_bytes, dtype=np.int16)
            
            # Run conversion
            audio_out = converter.convert(audio_in)
            
            # Write to output stream
            output_stream.write(audio_out.tobytes())
            
            # Report latency
            latency = converter.get_latency()
            latencies.append(latency)
            if len(latencies) > 100:
                latencies.pop(0)
                
            # Log average latency every 3 seconds
            if len(latencies) % 50 == 0:
                print(f"[Live] Avg Latency: {np.mean(latencies):.1f}ms | Max Latency: {np.max(latencies):.1f}ms", end="\r")
                
    except KeyboardInterrupt:
        print("\nStopping voice changer...")
    finally:
        # Clean up streams
        input_stream.stop_stream()
        input_stream.close()
        output_stream.stop_stream()
        output_stream.close()
        audio.terminate()
        print("Audio streams closed. Bye!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-Talk Standalone Inference Client (CPU)")
    parser.add_argument("--list_devices", action="store_true", help="List all available audio input/output devices")
    parser.add_argument("--model", type=str, default="distilled_store/distilled_voice.onnx", help="Path to distilled .onnx model")
    parser.add_argument("--vocos", type=str, default="distilled_store/vocos.onnx", help="Path to vocos .onnx model")
    parser.add_argument("--input_device", type=int, default=-1, help="PyAudio input device index (microhpone)")
    parser.add_argument("--output_device", type=int, default=-1, help="PyAudio output device index (virtual cable)")
    parser.add_argument("--chunk_size", type=int, default=512, help="Buffer size in samples (default: 512 = 21.3ms at 24kHz)")
    
    args = parser.parse_args()
    
    if args.list_devices:
        list_audio_devices()
        sys.exit(0)
        
    # Verify models exist
    if not os.path.exists(args.model):
        print(f"Error: Model not found at {args.model}")
        print("Please run distillation in the web dashboard first or provide the correct --model path.")
        sys.exit(1)
        
    if not os.path.exists(args.vocos):
        # Look in current folder as fallback
        if os.path.exists("vocos.onnx"):
            args.vocos = "vocos.onnx"
        else:
            print(f"Error: Vocos model not found at {args.vocos}")
            print("Please export Vocos in the web dashboard first or provide the correct --vocos path.")
            sys.exit(1)
            
    # Check if user needs device lists
    if args.input_device == -1 or args.output_device == -1:
        print("Warning: Input/Output device indexes are not specified.")
        print("Listing available devices to help you choose:")
        list_audio_devices()
        print("Run the command with device indexes:")
        print(f"python src/inference_cli.py --model {args.model} --vocos {args.vocos} --input_device <MIC_INDEX> --output_device <CABLE_INDEX>")
        sys.exit(0)
        
    run_inference_loop(
        model_path=args.model,
        vocos_path=args.vocos,
        input_idx=args.input_device,
        output_idx=args.output_device,
        chunk_size=args.chunk_size
    )
