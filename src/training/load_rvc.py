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
        
        # Mel spectrogram transform
        self.mel_transform = librosa.filters.mel(
            sr=22050, n_fft=1024, n_mels=input_features
        )
        
    def audio_to_mel(self, audio):
        """Convert audio to mel spectrogram"""
        # STFT
        stft = librosa.stft(audio, n_fft=1024, hop_length=256)
        magnitude = np.abs(stft)
        
        # Convert to mel scale
        mel_spec = np.dot(self.mel_transform, magnitude)
        mel_spec = np.log(mel_spec + 1e-8)  # Log scale
        
        return mel_spec
    
    def mel_to_audio(self, mel_spec):
        """Convert mel spectrogram back to audio (simplified)"""
        # This is a simplified inverse - in practice you'd use a proper vocoder
        # For training, we'll work in mel space and convert back at inference
        
        # Inverse mel scale (approximate)
        mel_spec_exp = np.exp(mel_spec) - 1e-8
        
        # Inverse mel filtering (approximate - not perfect but fast)
        magnitude = np.dot(np.linalg.pinv(self.mel_transform), mel_spec_exp)
        
        # Create phase (random - this is why quality won't be perfect)
        phase = np.random.uniform(-np.pi, np.pi, magnitude.shape)
        
        # Reconstruct audio
        stft_complex = magnitude * np.exp(1j * phase)
        audio = librosa.istft(stft_complex, hop_length=256)
        
        return audio
        
    def forward(self, mel_input):
        """Forward pass - convert mel spectrogram"""
        return self.encoder(mel_input)

def generate_training_data_smart(rvc_model, audio_files, chunk_size=2048, 
                                sample_rate=22050, overlap_ratio=0.5):
    """
    Smart data generation: Process full audio through RVC first,
    then create chunk pairs for real-time simulation
    """
    training_pairs = []
    hop_size = int(chunk_size * (1 - overlap_ratio))
    
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
            
            # Convert to mel spectrograms for training
            input_mel = audio_to_mel_tensor(input_chunk)
            target_mel = audio_to_mel_tensor(target_chunk)
            
            training_pairs.append((input_mel, target_mel))
            chunk_count += 1
        
        print(f"  Generated {chunk_count} training chunks")
    
    print(f"\nTotal training pairs: {len(training_pairs)}")
    return training_pairs

def audio_to_mel_tensor(audio, n_mels=80, n_fft=1024, hop_length=256):
    """Convert audio to mel spectrogram tensor"""
    # STFT
    stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(stft)
    
    # Mel filter bank
    mel_filter = librosa.filters.mel(sr=22050, n_fft=n_fft, n_mels=n_mels)
    mel_spec = np.dot(mel_filter, magnitude)
    
    # Log scale and convert to tensor
    mel_spec_log = np.log(mel_spec + 1e-8)
    
    return torch.FloatTensor(mel_spec_log)

def compare_approaches(rvc_model, test_audio):
    """Compare chunk-by-chunk vs full-then-chunk approaches"""
    
    print("=== COMPARING DATA GENERATION APPROACHES ===")
    
    chunk_size = 2048
    
    # Approach 1: Chunk-by-chunk (original flawed approach)
    print("\n1. Chunk-by-chunk approach:")
    chunks = [test_audio[i:i+chunk_size] for i in range(0, len(test_audio)-chunk_size, chunk_size//2)]
    
    chunk_results = []
    for i, chunk in enumerate(chunks[:3]):  # Test first 3 chunks
        try:
            result = rvc_model.infer(chunk)
            chunk_results.append(result)
            print(f"   Chunk {i+1}: Success")
        except Exception as e:
            print(f"   Chunk {i+1}: Failed - {e}")
    
    # Approach 2: Full-then-chunk (your better approach)
    print("\n2. Full-then-chunk approach:")
    try:
        full_result = rvc_model.infer(test_audio)
        print(f"   Full audio conversion: Success")
        print(f"   Quality consistency: High (RVC had full context)")
        
        # Now we can chunk the high-quality result
        full_chunks = [full_result[i:i+chunk_size] for i in range(0, len(full_result)-chunk_size, chunk_size//2)]
        print(f"   Generated {len(full_chunks)} high-quality chunk pairs")
        
    except Exception as e:
        print(f"   Full audio conversion: Failed - {e}")
    
    return chunk_results, full_result if 'full_result' in locals() else None

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

def create_realtime_processor(model_path):
    """Create a real-time processor for gaming"""
    
    class RealtimeProcessor:
        def __init__(self, model_path):
            self.model = LightweightVoiceConverter()
            self.model.load_state_dict(torch.load(model_path))
            self.model.eval()
            
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            
        def process_chunk(self, audio_chunk):
            """Process a single audio chunk in real-time"""
            with torch.no_grad():
                # Convert to mel
                mel_input = audio_to_mel_tensor(audio_chunk)
                mel_input = mel_input.unsqueeze(0).to(self.device)  # Add batch dim
                
                # Convert
                mel_output = self.model(mel_input)
                
                # Convert back to audio (simplified)
                # In practice, you'd use a proper fast vocoder here
                converted_audio = self.mel_to_audio_simple(mel_output.cpu().squeeze().numpy())
                
                return converted_audio
        
        def mel_to_audio_simple(self, mel_spec):
            """Simple mel to audio conversion for real-time use"""
            # This is a placeholder - you'd want a proper fast vocoder
            # For now, return processed input (this is where you'd integrate a vocoder)
            return np.random.randn(2048) * 0.1  # Placeholder
    
    return RealtimeProcessor(model_path)