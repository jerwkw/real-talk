import os
import sys
import pytest
import numpy as np
import torch
from pathlib import Path

# Ensure src directory is in path
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from training.distill import StudentVoiceConverter
from models.models import DistilledONNXConverter

def test_student_model_shapes():
    """Verify that the causal CNN + GRU model runs and maintains correct shape boundaries"""
    model = StudentVoiceConverter(mel_channels=100, hidden_dim=128, num_gru_layers=1)
    
    # [Batch, Channels, Time]
    dummy_mel = torch.randn(2, 100, 64)
    dummy_f0 = torch.randn(2, 1, 64)
    
    out = model(dummy_mel, dummy_f0)
    
    # Expected output: [Batch, Channels, Time]
    assert out.shape == (2, 100, 64), f"Expected shape (2, 100, 64), got {out.shape}"

def test_onnx_export_and_load(tmp_path):
    """Verify student model exports to ONNX successfully and loads in ONNX Runtime"""
    import onnxruntime as ort
    
    model = StudentVoiceConverter(mel_channels=100, hidden_dim=64, num_gru_layers=1)
    model.eval()
    
    onnx_file = tmp_path / "test_student.onnx"
    
    dummy_mel = torch.randn(1, 100, 32)
    dummy_f0 = torch.randn(1, 1, 32)
    
    # Export
    torch.onnx.export(
        model,
        (dummy_mel, dummy_f0),
        str(onnx_file),
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["mel", "f0"],
        output_names=["converted_mel"],
        dynamic_axes={
            "mel": {0: "batch_size", 2: "time_frames"},
            "f0": {0: "batch_size", 2: "time_frames"},
            "converted_mel": {0: "batch_size", 2: "time_frames"}
        },
        dynamo=False
    )
    
    assert onnx_file.exists(), "ONNX file was not created"
    
    # Load with ONNX Runtime
    session = ort.InferenceSession(str(onnx_file), providers=["CPUExecutionProvider"])
    assert session is not None, "Failed to load ONNX session"
    
    input_names = [i.name for i in session.get_inputs()]
    assert "mel" in input_names
    assert "f0" in input_names
    
    # Run mock inference
    mel_in = np.random.randn(1, 100, 32).astype(np.float32)
    f0_in = np.random.randn(1, 1, 32).astype(np.float32)
    
    outputs = session.run(None, {"mel": mel_in, "f0": f0_in})
    assert outputs[0].shape == (1, 100, 32), f"Expected shape (1, 100, 32), got {outputs[0].shape}"
