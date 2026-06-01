import torch
import torch.nn as nn
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path

class LightweightVoiceConverter(nn.Module):
    """Ultra-fast voice converter for real-time gaming"""
    def __init__(self, input_features=80, hidden_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(input_features, hidden_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, input_features, 3, padding=1)
        )
        
    def forward(self, mel_input):
        """Forward pass - convert mel spectrogram"""
        return self.encoder(mel_input)

class RealtimeAudioProcessor:
    """Handles audio ↔ mel conversion and real-time processing"""
    def __init__(self, model, sample_rate=44100, n_fft=1024, hop_length=256, n_mels=80):
        self.model = model
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        
        # Create mel filter bank
        self.mel_filter = librosa.filters.mel(
            sr=sample_rate, n_fft=n_fft, n_mels=n_mels
        )
        
        # For audio reconstruction (you'd use a proper vocoder here)
        self.inv_mel_filter = np.linalg.pinv(self.mel_filter)
        
    def audio_to_mel(self, audio):
        """Convert audio to mel spectrogram tensor"""
        # STFT
        stft = librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length)
        magnitude = np.abs(stft)
        
        # Convert to mel scale
        mel_spec = np.dot(self.mel_filter, magnitude)
        mel_spec_log = np.log(mel_spec + 1e-8)
        
        return torch.FloatTensor(mel_spec_log)
    
    def mel_to_audio(self, mel_tensor):
        """Convert mel spectrogram back to audio (simplified)"""
        mel_spec = mel_tensor.detach().cpu().numpy()
        
        # Inverse log
        mel_spec_exp = np.exp(mel_spec) - 1e-8
        
        # Approximate inverse mel filtering
        magnitude = np.dot(self.inv_mel_filter, mel_spec_exp)
        
        # Random phase (this is why we need a proper vocoder for quality)
        phase = np.random.uniform(-np.pi, np.pi, magnitude.shape)
        
        # Reconstruct
        stft_complex = magnitude * np.exp(1j * phase)
        audio = librosa.istft(stft_complex, hop_length=self.hop_length)
        
        return audio
    
    def process_realtime(self, audio_chunk):
        """Process audio chunk in real-time"""
        with torch.no_grad():
            # Audio → Mel
            mel_input = self.audio_to_mel(audio_chunk)
            mel_input = mel_input.unsqueeze(0)  # Add batch dimension
            
            # Mel → Converted Mel
            mel_output = self.model(mel_input)
            
            # Converted Mel → Audio
            converted_audio = self.mel_to_audio(mel_output.squeeze(0))
            
            return converted_audio

def generate_training_data_smart(rvc_model, audio_files, chunk_size=2048, 
                                sample_rate=44100, overlap_ratio=0.5):
    """
    Process full audio through RVC first,
    then create chunk pairs for real-time simulation
    """
    training_pairs = []
    hop_size = int(chunk_size * (1 - overlap_ratio))
    
    # Create processor for consistent mel conversion
    temp_model = LightweightVoiceConverter()  # Temporary model for processor
    processor = RealtimeAudioProcessor(temp_model, sample_rate=sample_rate)
    
    print(f"Generating training data with {chunk_size} samples per chunk...")
    
    for i, audio_file in enumerate(audio_files):
        print(f"Processing file {i+1}/{len(audio_files)}: {audio_file}")
        
        # Load full audio file
        original_audio, sr = librosa.load(audio_file, sr=sample_rate)
        
        # Skip if audio is too short
        if len(original_audio) < chunk_size * 2:
            print(f"  Skipping {audio_file} - too short")
            continue
            
        print(f"  Original audio length: {len(original_audio)/sample_rate:.1f}s")
        
        # ✅ SMART APPROACH: Process FULL audio through RVC first
        print("  Converting full audio through RVC...")
        try:
            # This is where you'd call your RVC model on the complete audio
            converted_audio_full = rvc_model.infer(original_audio)
            print(f"  RVC conversion successful!")
            
        except Exception as e:
            print(f"  RVC conversion failed: {e}")
            continue
        
        # Ensure same length (handle any size differences from RVC)
        min_len = min(len(original_audio), len(converted_audio_full))
        original_audio = original_audio[:min_len]
        converted_audio_full = converted_audio_full[:min_len]
        
        # Now create chunk pairs from the high-quality full conversion
        chunk_count = 0
        for start_idx in range(0, len(original_audio) - chunk_size, hop_size):
            end_idx = start_idx + chunk_size
            
            # Extract corresponding chunks
            input_chunk = original_audio[start_idx:end_idx]
            target_chunk = converted_audio_full[start_idx:end_idx]
            
            # Convert to mel spectrograms using RealtimeAudioProcessor for consistency
            input_mel = processor.audio_to_mel(input_chunk)
            target_mel = processor.audio_to_mel(target_chunk)
            
            training_pairs.append((input_mel, target_mel))
            chunk_count += 1
        
        print(f"  Generated {chunk_count} training chunks")
    
    print(f"\nTotal training pairs: {len(training_pairs)}")
    return training_pairs

def train_lightweight_model(training_pairs, epochs=50, lr=0.001, batch_size=16):
    """Train the lightweight model on the smart-generated data"""
    
    model = LightweightVoiceConverter()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    # Create data loader
    dataset = torch.utils.data.TensorDataset(
        torch.stack([pair[0] for pair in training_pairs]),
        torch.stack([pair[1] for pair in training_pairs])
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    print(f"Training on {len(training_pairs)} samples for {epochs} epochs...")
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        batch_count = 0
        
        for batch_input, batch_target in dataloader:
            # Forward pass
            output = model(batch_input)
            
            # Handle size mismatches
            min_seq = min(output.shape[-1], batch_target.shape[-1])
            output = output[..., :min_seq]
            batch_target = batch_target[..., :min_seq]
            
            # Calculate loss
            loss = criterion(output, batch_target)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            batch_count += 1
        
        avg_loss = total_loss / batch_count
        print(f"Epoch {epoch+1:3d}/{epochs}: Loss = {avg_loss:.6f}")
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"lightweight_model_epoch_{epoch+1}.pth")
    
    print("Training completed!")
    return model
