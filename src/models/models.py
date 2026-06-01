"""This file contains the abstract base class for voice conversion models and their implementations."""

from abc import ABC, abstractmethod
import numpy as np
import time
import os
from typing import Dict, Any

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

class DummyConverter(VoiceConverter):
    """Dummy voice conversion model for testing purposes"""
    
    def __init__(self):
        super().__init__("Dummy")
        
    def load_model(self, model_path: str, **kwargs):
        self.is_loaded = True
        self.model_info = {
            "type": "Dummy",
            "model_path": model_path,
            "sample_rate": kwargs.get("sample_rate", 24000)
        }
        return True
        
    def convert(self, audio_input: np.ndarray) -> np.ndarray:
        return audio_input
        
    def unload_model(self):
        self.is_loaded = False

class RVCConverter(VoiceConverter):
    """RVC (Retrieval-based Voice Conversion) implementation using rvc-python"""
    
    def __init__(self):
        super().__init__("RVC")
        self.model = None
        
    def load_model(self, model_path: str, **kwargs):
        try:
            from rvc_python.infer import RVCInference
            import torch
            
            device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
            print(f"Loading RVC model from {model_path} on device {device}...")
            
            self.model = RVCInference(device=device)
            self.model.load_model(model_path)
            
            self.is_loaded = True
            self.model_info = {
                "type": "RVC",
                "model_path": model_path,
                "device": device,
                "sample_rate": kwargs.get("sample_rate", 24000)
            }
            return True
        except Exception as e:
            print(f"Failed to load RVC model: {e}")
            return False
            
    def convert(self, audio_input: np.ndarray) -> np.ndarray:
        if not self.is_loaded or self.model is None:
            return audio_input
            
        start_time = time.time()
        import tempfile
        import soundfile as sf
        
        try:
            # RVC Inference runs on files, so route through temp files
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_in:
                input_path = temp_in.name
                sf.write(input_path, audio_input, 24000)
                
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_out:
                output_path = temp_out.name
            
            # Run inference
            self.model.infer_file(input_path, output_path)
            
            # Read converted audio
            converted_audio, _ = sf.read(output_path)
            
            # Clean up
            os.remove(input_path)
            os.remove(output_path)
            
            self.latency_ms = (time.time() - start_time) * 1000
            return converted_audio
            
        except Exception as e:
            print(f"RVC Convert error: {e}")
            self.latency_ms = (time.time() - start_time) * 1000
            return audio_input
        
    def unload_model(self):
        self.model = None
        self.is_loaded = False

class DistilledONNXConverter(VoiceConverter):
    """Distilled Voice conversion running mapping and Vocos in ONNX Runtime on CPU"""
    
    def __init__(self):
        super().__init__("Distilled ONNX")
        self.mapping_session = None
        self.vocos = None
        
    def load_model(self, model_path: str, **kwargs):
        import onnxruntime as ort
        from models.vocos_onnx import ONNXVocos
        
        try:
            print(f"Loading distilled mapping model from: {model_path}")
            self.mapping_session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
            self.input_names = [i.name for i in self.mapping_session.get_inputs()]
            
            vocos_path = kwargs.get("vocos_onnx_path", "vocos.onnx")
            if not os.path.exists(vocos_path):
                # Search in same folder
                model_dir = os.path.dirname(model_path)
                vocos_path = os.path.join(model_dir, "vocos.onnx")
                
            if not os.path.exists(vocos_path):
                # Check current directory
                vocos_path = "vocos.onnx"
                
            print(f"Loading Vocos ONNX from: {vocos_path}")
            self.vocos = ONNXVocos(vocos_path)
            
            self.is_loaded = True
            self.model_info = {
                "type": "Distilled ONNX",
                "mapping_path": model_path,
                "vocos_path": vocos_path,
                "sample_rate": kwargs.get("sample_rate", 24000)
            }
            return True
        except Exception as e:
            print(f"Failed to load Distilled ONNX model: {e}")
            return False
            
    def convert(self, audio_input: np.ndarray) -> np.ndarray:
        if not self.is_loaded or self.mapping_session is None or self.vocos is None:
            return audio_input
            
        start_time = time.time()
        from training.distill import compute_mel_spectrogram, extract_f0
        
        try:
            # 1. Ensure float32 representation
            if audio_input.dtype == np.int16:
                audio_float = audio_input.astype(np.float32) / 32768.0
            else:
                audio_float = audio_input.astype(np.float32)
                
            # 2. Extract Mel and F0
            # For real-time chunk conversion, sample rate is 24kHz
            mel = compute_mel_spectrogram(audio_float, sr=24000) # [100, T]
            f0 = extract_f0(audio_float, sr=24000) # [T]
            
            # Add batch/channel dimensions
            mel_input = np.expand_dims(mel, axis=0).astype(np.float32) # [1, 100, T]
            f0_input = np.expand_dims(np.expand_dims(f0, axis=0), axis=0).astype(np.float32) # [1, 1, T]
            
            # 3. Predict target Mel using the distilled ONNX model
            outputs = self.mapping_session.run(None, {
                self.input_names[0]: mel_input,
                self.input_names[1]: f0_input
            })
            pred_mel = outputs[0] # [1, 100, T]
            
            # 4. Decode target Mel to audio using Vocos ONNX
            audio_out = self.vocos.decode(pred_mel) # [Samples,]
            
            # 5. Convert back to original input format (int16 or float32)
            if audio_input.dtype == np.int16:
                audio_out = np.clip(audio_out * 32768.0, -32768, 32767).astype(np.int16)
            
            self.latency_ms = (time.time() - start_time) * 1000
            return audio_out
            
        except Exception as e:
            print(f"Distilled conversion error: {e}")
            self.latency_ms = (time.time() - start_time) * 1000
            return audio_input
            
    def unload_model(self):
        self.mapping_session = None
        self.vocos = None
        self.is_loaded = False

class BitNetConverter(VoiceConverter):
    """BitNet-based voice conversion implementation (placeholder)"""
    
    def __init__(self):
        super().__init__("BitNet")
        
    def load_model(self, model_path: str, **kwargs):
        self.is_loaded = True
        self.model_info = {
            "type": "BitNet",
            "model_path": model_path
        }
        return True
        
    def convert(self, audio_input: np.ndarray) -> np.ndarray:
        return audio_input
        
    def unload_model(self):
        self.is_loaded = False

class TraditionalConverter(VoiceConverter):
    """Traditional DSP-based voice conversion (pitch shifting)"""
    
    def __init__(self):
        super().__init__("Traditional DSP")
        self.pitch_shift = 1.0
        
    def load_model(self, model_path: str, **kwargs):
        self.pitch_shift = kwargs.get("pitch_shift", 1.2) # Default shift up
        self.is_loaded = True
        self.model_info = {
            "type": "Traditional DSP",
            "pitch_shift": self.pitch_shift
        }
        return True
        
    def convert(self, audio_input: np.ndarray) -> np.ndarray:
        start_time = time.time()
        
        if self.pitch_shift != 1.0:
            # Pitch shift by crude resampling
            new_length = int(len(audio_input) / self.pitch_shift)
            converted = np.interp(
                np.linspace(0, len(audio_input) - 1, new_length),
                np.arange(len(audio_input)),
                audio_input
            )
            if audio_input.dtype == np.int16:
                converted_audio = converted.astype(np.int16)
            else:
                converted_audio = converted.astype(np.float32)
        else:
            converted_audio = audio_input
            
        self.latency_ms = (time.time() - start_time) * 1000
        return converted_audio
        
    def unload_model(self):
        self.is_loaded = False