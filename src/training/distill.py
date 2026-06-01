import os
import sys
import zipfile
import urllib.request
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import librosa
from pathlib import Path

# Set up path to import other local modules if needed
sys.path.append(str(Path(__file__).resolve().parent.parent))

class CausalConv1d(nn.Module):
    """1D Convolution with causal padding to prevent looking into the future"""
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation)

    def forward(self, x):
        # x shape: [B, C, T]
        # Pad on the left side of the time dimension
        x = F.pad(x, (self.padding, 0))
        return self.conv(x)

class StudentVoiceConverter(nn.Module):
    """Lightweight Causal CNN + GRU model for Spectrogram-to-Spectrogram conversion"""
    def __init__(self, mel_channels=100, hidden_dim=256, num_gru_layers=2):
        super().__init__()
        # Input: mel_channels + 1 (F0)
        self.input_layer = CausalConv1d(mel_channels + 1, hidden_dim, kernel_size=3)
        
        self.conv_blocks = nn.Sequential(
            CausalConv1d(hidden_dim, hidden_dim, kernel_size=3, dilation=2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            CausalConv1d(hidden_dim, hidden_dim, kernel_size=3, dilation=4),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
        )
        
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_gru_layers,
            batch_first=True,
            bidirectional=False # Unidirectional for real-time streaming
        )
        
        self.output_layer = nn.Linear(hidden_dim, mel_channels)
        
    def forward(self, mel, f0):
        # mel: [B, C, T], f0: [B, 1, T]
        x = torch.cat([mel, f0], dim=1) # [B, C + 1, T]
        
        x = self.input_layer(x)
        x = self.conv_blocks(x) # [B, hidden_dim, T]
        
        # GRU expects sequence shape: [B, T, hidden_dim]
        x = x.transpose(1, 2)
        x, _ = self.gru(x) # [B, T, hidden_dim]
        
        # Map back to mel dimension
        out = self.output_layer(x) # [B, T, C]
        
        # Transpose back to [B, C, T] for Vocos Vocoder
        out = out.transpose(1, 2)
        return out

def get_best_device():
    """Select the best available acceleration hardware"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def download_default_dataset(dest_dir: str) -> list:
    """
    Downloads the Free Spoken Digit Dataset (FSDD) as a default training source.
     FSDD is small (~10MB) and has multiple speakers (Jackson, Nicolas, Yamil, etc.).
    """
    dest_path = Path(dest_dir)
    dest_path.mkdir(parents=True, exist_ok=True)
    
    zip_url = "https://github.com/Jakobovski/free-spoken-digit-dataset/archive/refs/heads/master.zip"
    zip_path = dest_path / "fsdd.zip"
    
    if not zip_path.exists():
        print(f"Downloading default dataset from {zip_url}...")
        urllib.request.urlretrieve(zip_url, zip_path)
        print("Download complete.")
        
    extracted_dir = dest_path / "free-spoken-digit-dataset-master"
    if not extracted_dir.exists():
        print("Extracting dataset...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(dest_path)
        print("Extraction complete.")
        
    wav_dir = extracted_dir / "recordings"
    wav_files = list(wav_dir.glob("*.wav"))
    print(f"Found {len(wav_files)} source wav files.")
    return [str(p) for p in wav_files]

def compute_mel_spectrogram(y, sr=24000, n_fft=1024, hop_length=256, n_mels=100):
    """Compute log-mel spectrogram matching Vocos vocoder spec"""
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, fmin=0, fmax=sr//2
    )
    log_S = np.log(np.clip(S, a_min=1e-5, a_max=None))
    return log_S

def extract_f0(y, sr=24000, hop_length=256, fmin=50, fmax=800):
    """Extract pitch F0 using librosa YIN algorithm and normalize it"""
    # Yin returns pitch per frame
    f0 = librosa.yin(y=y, sr=sr, fmin=fmin, fmax=fmax, hop_length=hop_length)
    
    # Replace NaNs or extreme values
    f0 = np.nan_to_num(f0)
    
    # Normalize log(f0) for voiced sections (voiced when F0 > fmin)
    mean_f0 = 150.0
    std_f0 = 50.0
    
    f0_norm = np.zeros_like(f0)
    voiced = f0 > fmin
    if np.any(voiced):
        f0_norm[voiced] = (np.log(f0[voiced]) - np.log(mean_f0)) / (std_f0 / mean_f0)
        
    return f0_norm

def prepare_distillation_data(rvc_model_path: str, rvc_index_path: str, wav_paths: list, 
                              output_dir: str, device_name: str = "cpu", limit: int = 150):
    """
    1. Runs RVC teacher model on source wav files to generate target character speech.
    2. Computes and aligns Mel-spectrograms and F0 curves.
    3. Saves aligned training pairs.
    """
    from rvc_python.infer import RVCInference
    
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Initializing RVC teacher model inference on device: {device_name}...")
    rvc = RVCInference(device=device_name)
    rvc.load_model(rvc_model_path)
    
    # Cap files count for faster distillation run if needed
    files_to_process = wav_paths[:limit]
    total_files = len(files_to_process)
    
    print(f"Processing {total_files} files through RVC teacher model...")
    pairs_count = 0
    
    for idx, wav_path in enumerate(files_to_process):
        try:
            # Load and resample source file
            y_src, sr = librosa.load(wav_path, sr=24000)
            
            # Temporary path for RVC output
            temp_target_path = out_path / f"temp_rvc_out_{idx}.wav"
            
            # Run RVC inference using CLI/infer wrapper
            # Use index file if available
            index_arg = rvc_index_path if rvc_index_path and os.path.exists(rvc_index_path) else None
            
            # We use rvc_python's infer_file or load_model + infer_file
            # Note: rvc.infer_file works on paths
            rvc.infer_file(wav_path, str(temp_target_path))
            
            # Load RVC output
            if not temp_target_path.exists():
                continue
                
            y_tgt, _ = librosa.load(str(temp_target_path), sr=24000)
            
            # Clean up temp file
            os.remove(temp_target_path)
            
            # Align lengths
            min_len = min(len(y_src), len(y_tgt))
            if min_len < 1024:
                continue
                
            y_src = y_src[:min_len]
            y_tgt = y_tgt[:min_len]
            
            # Compute input features (mel + f0)
            src_mel = compute_mel_spectrogram(y_src)
            src_f0 = extract_f0(y_src)
            
            # Compute target features (mel)
            tgt_mel = compute_mel_spectrogram(y_tgt)
            
            # Ensure time frame sizes match exactly
            min_frames = min(src_mel.shape[1], tgt_mel.shape[1], len(src_f0))
            if min_frames < 4:
                continue
                
            src_mel = src_mel[:, :min_frames]
            src_f0 = src_f0[:min_frames]
            tgt_mel = tgt_mel[:, :min_frames]
            
            # Save preprocessed features
            np.save(out_path / f"src_mel_{pairs_count}.npy", src_mel)
            np.save(out_path / f"src_f0_{pairs_count}.npy", src_f0)
            np.save(out_path / f"tgt_mel_{pairs_count}.npy", tgt_mel)
            
            pairs_count += 1
            if pairs_count % 10 == 0 or pairs_count == total_files:
                print(f" Prepared {pairs_count}/{total_files} training pairs...")
                
        except Exception as e:
            print(f"Error processing {wav_path}: {e}")
            
    print(f"Data preparation complete! Created {pairs_count} feature files.")
    return pairs_count

class DistillationDataset(torch.utils.data.Dataset):
    """Loads pre-saved numpy features for PyTorch training"""
    def __init__(self, data_dir: str, num_pairs: int, max_frames: int = 256):
        self.data_dir = Path(data_dir)
        self.num_pairs = num_pairs
        self.max_frames = max_frames
        
    def __len__(self):
        return self.num_pairs
        
    def __getitem__(self, idx):
        src_mel = np.load(self.data_dir / f"src_mel_{idx}.npy")
        src_f0 = np.load(self.data_dir / f"src_f0_{idx}.npy")
        tgt_mel = np.load(self.data_dir / f"tgt_mel_{idx}.npy")
        
        # Crop or pad to max_frames for uniform batches
        t = src_mel.shape[1]
        if t > self.max_frames:
            # Random slice during training
            start = np.random.randint(0, t - self.max_frames)
            src_mel = src_mel[:, start:start+self.max_frames]
            src_f0 = src_f0[start:start+self.max_frames]
            tgt_mel = tgt_mel[:, start:start+self.max_frames]
        else:
            # Zero pad
            pad_len = self.max_frames - t
            src_mel = np.pad(src_mel, ((0,0), (0, pad_len)))
            src_f0 = np.pad(src_f0, (0, pad_len))
            tgt_mel = np.pad(tgt_mel, ((0,0), (0, pad_len)))
            
        return (
            torch.FloatTensor(src_mel), 
            torch.FloatTensor(src_f0).unsqueeze(0), # Add feature dimension -> (1, T)
            torch.FloatTensor(tgt_mel)
        )

def run_distillation_training(data_dir: str, num_pairs: int, output_onnx_path: str, 
                              epochs: int = 40, batch_size: int = 16, lr: float = 1e-3, 
                              progress_callback=None):
    """Trains the Causal CNN + GRU student model and exports to ONNX"""
    device = get_best_device()
    print(f"Training student model on device: {device}")
    
    dataset = DistillationDataset(data_dir, num_pairs)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = StudentVoiceConverter(mel_channels=100, hidden_dim=256, num_gru_layers=2)
    model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        batches = 0
        
        for src_mel, src_f0, tgt_mel in dataloader:
            src_mel = src_mel.to(device)
            src_f0 = src_f0.to(device)
            tgt_mel = tgt_mel.to(device)
            
            # Forward pass
            pred_mel = model(src_mel, src_f0)
            
            # Compute loss
            loss = criterion(pred_mel, tgt_mel)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            batches += 1
            
        avg_loss = total_loss / batches if batches > 0 else 0.0
        log_msg = f"Epoch {epoch+1:03d}/{epochs:03d} - Loss: {avg_loss:.6f}"
        print(log_msg)
        
        if progress_callback:
            progress_callback(epoch + 1, epochs, avg_loss)
            
    # Export trained model to ONNX
    print("Training finished. Exporting student model to ONNX...")
    model.eval()
    model.to(torch.device("cpu"))
    
    dummy_mel = torch.randn(1, 100, 128)
    dummy_f0 = torch.randn(1, 1, 128)
    
    torch.onnx.export(
        model,
        (dummy_mel, dummy_f0),
        output_onnx_path,
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
    print(f"Student model exported to ONNX: {output_onnx_path}")
    return model

if __name__ == "__main__":
    # Test script standalone run
    import argparse
    parser = argparse.ArgumentParser(description="Distill RVC Voice Model to ONNX")
    parser.add_argument("--rvc_model", type=str, required=True, help="Path to RVC teacher model .pth file")
    parser.add_argument("--rvc_index", type=str, default="", help="Path to RVC teacher model .index file")
    parser.add_argument("--output_onnx", type=str, default="distilled_voice.onnx", help="Output path for distilled model .onnx")
    parser.add_argument("--custom_source_dir", type=str, default="", help="Optional folder containing user's custom source WAV files")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--device", type=str, default="cpu", help="Device for RVC teacher running (cpu, cuda, mps)")
    args = parser.parse_args()
    
    # Define directories
    temp_dir = "./temp_distill"
    os.makedirs(temp_dir, exist_ok=True)
    
    # 1. Get Source WAV files
    if args.custom_source_dir and os.path.exists(args.custom_source_dir):
        print(f"Using custom source directory: {args.custom_source_dir}")
        wavs = list(Path(args.custom_source_dir).glob("*.wav"))
        wav_files = [str(p) for p in wavs]
        if not wav_files:
            print("No WAV files found in custom directory. Falling back to default FSDD dataset.")
            wav_files = download_default_dataset(temp_dir)
    else:
        wav_files = download_default_dataset(temp_dir)
        
    # 2. Run RVC teacher and prepare features
    pairs_num = prepare_distillation_data(
        rvc_model_path=args.rvc_model,
        rvc_index_path=args.rvc_index,
        wav_paths=wav_files,
        output_dir=temp_dir,
        device_name=args.device,
        limit=200
    )
    
    if pairs_num == 0:
        print("Failed to prepare training pairs. Exiting.")
        sys.exit(1)
        
    # 3. Train student model & export
    run_distillation_training(
        data_dir=temp_dir,
        num_pairs=pairs_num,
        output_onnx_path=args.output_onnx,
        epochs=args.epochs,
        batch_size=8
    )
