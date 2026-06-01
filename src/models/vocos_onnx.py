import os
import torch
import torch.nn as nn
import numpy as np
from typing import Optional

# We'll import vocos inside functions so that the CLI doesn't strictly depend on having the full PyTorch/Vocos installed for CPU inference
class VocosONNXWrapper(nn.Module):
    """Wrapper module for exporting Vocos to ONNX"""
    def __init__(self, vocos_model):
        super().__init__()
        self.vocos = vocos_model
        
    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        # Pretrained Vocos decode method:
        # In vocos, decode takes a log-mel spectrogram of shape (B, C, T) and returns audio (B, T)
        return self.vocos.decode(mel)

def export_vocos_to_onnx(onnx_path: str, model_name: str = "charactr/vocos-mel-24khz"):
    """
    Loads pretrained Vocos model and exports it to ONNX format.
    Run this in the training environment (GPU/WSL/Mac).
    """
    from vocos import Vocos
    
    print(f"Loading pretrained Vocos model: {model_name}")
    vocos = Vocos.from_pretrained(model_name)
    vocos.eval()
    
    wrapper = VocosONNXWrapper(vocos)
    wrapper.eval()
    
    # Vocos mel model expects 100 mel channels
    # Dummy input format: [Batch, Channels, Time_Frames]
    # We use a dummy sequence length of 256 frames
    dummy_input = torch.randn(1, 100, 256)
    
    print(f"Exporting Vocos to {onnx_path}...")
    torch.onnx.export(
        wrapper,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["mel"],
        output_names=["audio"],
        dynamic_axes={
            "mel": {0: "batch_size", 2: "time_frames"},
            "audio": {0: "batch_size", 1: "samples"}
        },
        dynamo=False
    )
    print("Vocos successfully exported to ONNX!")

class ONNXVocos:
    """Runs Vocos vocoder inference using ONNX Runtime on CPU"""
    def __init__(self, onnx_path: str):
        import onnxruntime as ort
        self.onnx_path = onnx_path
        
        # CPU execution provider for light inference
        self.session = ort.InferenceSession(
            onnx_path, 
            providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        
    def decode(self, mel_spec: np.ndarray) -> np.ndarray:
        """
        Reconstruct waveform from Mel spectrogram.
        Args:
            mel_spec: numpy array of shape (Channels, Time) or (1, Channels, Time)
        Returns:
            waveform: numpy array of shape (Samples,)
        """
        if len(mel_spec.shape) == 2:
            mel_spec = np.expand_dims(mel_spec, axis=0) # Add batch dimension -> (1, Channels, Time)
            
        # Ensure float32
        mel_spec = mel_spec.astype(np.float32)
        
        # Run ONNX inference
        outputs = self.session.run(None, {self.input_name: mel_spec})
        audio = outputs[0] # (1, Samples)
        
        # Squeeze batch dimension and return
        return audio[0]
