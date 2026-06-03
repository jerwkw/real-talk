# Real-Talk: Voice Distillation & Inference Framework

Real-Talk is a high-performance voice conversion framework designed for ultra-low latency real-time voice conversion during active gaming and live streaming. 

It implements a **Teacher-Student Knowledge Distillation** architecture that takes complex, heavy Retrieval-based Voice Conversion (RVC) models and distills them into a lightweight, CPU-friendly student model capable of running in real-time on standard CPUs (~20-30ms latency) without competing for gaming GPU resources.

---

## 📐 System Design & Architecture

The framework consists of two main execution phases: **Model Distillation (GPU-accelerated)** and **Real-Time Inference (CPU-optimized)**.

```
                  ┌──────────────────────────────────────────────┐
                  │          DISTILLATION PHASE (GPU)            │
                  │                                              │
                  │  ┌──────────────┐      ┌────────────────┐    │
                  │  │ Source Audio │ ───> │  RVC Teacher   │    │
                  │  └──────────────┘      └────────────────┘    │
                  │          │                      │            │
                  │          ▼                      ▼            │
                  │  ┌──────────────┐      ┌────────────────┐    │
                  │  │  Source Mel  │      │   Target Mel   │    │
                  │  └──────────────┘      └────────────────┘    │
                  │          │                      │            │
                  │          ▼                      ▼            │
                  │  ┌──────────────────────────────────────┐    │
                  │  │        Causal CNN-GRU Student        │    │
                  │  └──────────────────────────────────────┘    │
                  └───────────────────┬──────────────────────────┘
                                      │
                                      ▼ [Export ONNX]
                  ┌──────────────────────────────────────────────┐
                  │          INFERENCE PHASE (CPU)               │
                  │                                              │
                  │ ┌───────────┐     ┌──────────────────────┐   │
                  │ │ Mic Input │ ──> │ ONNX Student Mapping │   │
                  │ └───────────┘     └──────────────────────┘   │
                  │                              │               │
                  │                              ▼ [Predicted Mel]
                  │                   ┌──────────────────────┐   │
                  │                   │  ONNX Vocos Decoder  │   │
                  │                   └──────────────────────┘   │
                  │                              │               │
                  │                              ▼               │
                  │                   ┌──────────────────────┐   │
                  │                   │ Virtual Audio Cable  │   │
                  │                   └──────────────────────┘   │
                  └──────────────────────────────────────────────┘
```

### 1. Teacher-Student Distillation
- **The Challenge**: Standard RVC runs an extremely heavy HuBERT content extractor (90M+ parameters), followed by index-retrieval matching and a large HiFi-GAN vocoder. This pipeline is too computationally expensive to run on a CPU and occupies too much VRAM on a GPU while gaming.
- **The Solution**: We bypass the HuBERT encoder entirely during inference. We train an ultra-lightweight **Causal CNN + Unidirectional GRU** student model to directly map the user's log-mel spectrogram features and fundamental frequency (F0) curves into the target speaker's log-mel spectrogram.
- **The Speedup**: Bypassing HuBERT reduces parameter size from 150M+ down to 1.5M, allowing the student mapping model to run in less than **1ms** on a single CPU thread.

### 2. Audio Reconstruction via ONNX Vocos
- Rather than using traditional phase vocoders (which sound robotic) or standard heavy HiFi-GAN generators, Real-Talk uses **Vocos** (a fast Fourier-based neural vocoder).
- Vocos generates spectral coefficients and reconstructs waveforms via Inverse Fast Fourier Transform (IFFT) in a single forward pass, providing near-studio quality speech on CPU.
- Both the student mapping model and the Vocos vocoder are exported to **ONNX** format for highly optimized C++ execution speeds using the ONNX Runtime CPU execution provider.

### 3. Dual-Interface Runtime Design
- **Web Dashboard**: An elegant glassmorphic interface built on FastAPI and WebSockets, facilitating RVC model downloading, training data alignment, background distillation visualization, and browser-side performance testing.
- **Standalone CLI Client**: A zero-overhead python script that runs outside of the web server. It opens raw PyAudio capture/playback streams natively on the host machine to route voice signals into virtual cables with minimal RAM and CPU footprint.

---

## 📂 Project Repository Layout

- `src/models/models.py`: Interface abstractions for voice converters. Implements the `DistilledONNXConverter` that orchestrates real-time audio chunk preprocessing, pitch extraction, mapping session execution, and Vocos decoding.
- `src/models/vocos_onnx.py`: Utility module to trace and export the pre-trained Vocos vocoder model to ONNX.
- `src/training/distill.py`: Training pipeline that auto-downloads source datasets (FSDD), runs RVC teacher inference, performs F0 tracking, aligns datasets, and trains/exports the student model.
- `src/web/app.py` & `static/index.html`: Web server and glassmorphic dashboard interface.
- `src/inference_cli.py`: Standalone command-line script for active gaming inference.
- `pyproject.toml` & `uv.lock`: Dependency locks. Set up to run Python 3.10 to maintain compatibility with legacy fairseq/RVC libraries on modern systems.

---

## 🚀 Usage Guide

### Step 1: Environment Setup
We use `uv` for lightning-fast and reproducible dependency management. Install `uv` if you haven't already:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
Initialize the Python environment in the repository folder:
```bash
# Create a Python 3.10 virtual environment (to support fairseq/RVC config layers)
uv venv --python 3.10
# Sync dependencies
uv sync
```
*Note for macOS users: PyAudio requires PortAudio. If installation fails, run `brew install portaudio` before running `uv sync`.*

### Step 2: Virtual Audio Cable Setup
To route your converted voice to games or applications like Discord, you must install a virtual audio driver:
- **Windows**: Download and install [VB-Audio Virtual Cable](https://vb-audio.com/Cable/). It exposes **CABLE Input** (playback) and **CABLE Output** (recording) devices.
- **macOS**: Download and install [BlackHole 2ch](https://github.com/ExistentialAudio/BlackHole). It exposes a unified **BlackHole 2ch** device.

### Step 3: Run the Web Dashboard & Distill a Model
Because training requires running RVC teacher inference on the source audio, it is highly recommended to run this step on **Windows WSL2** (with CUDA GPU forwarding enabled) or on a **macOS** machine.

1. Launch the web dashboard server:
   ```bash
   uv run uvicorn src.web.app:app --host 0.0.0.0 --port 8000
   ```
2. Open `http://localhost:8000` in your web browser.
3. Paste a direct download URL to an RVC model `.pth` (from repositories like [voice-models.com](https://voice-models.com/)), name the file (e.g. `spongebob.pth`), and click **Download Model**.
4. Click **Export Vocos** to generate the vocoder ONNX file.
5. Under **Distillation Console**, select the RVC model, configure training options (e.g., 30-50 epochs), and click **Start Distillation**.
6. The panel will stream loss updates and training logs via WebSockets. Once finished, your distilled model will be saved to `distilled_store/<model_name>_distilled.onnx`.

### Step 4: Run Low-Latency Standalone Inference (For Gaming)
Once your voice is distilled, run the inference script **natively** on your host OS (Windows Cmd/macOS Terminal) for zero-overhead performance:

1. List available audio devices to identify your microphone and virtual cable index numbers:
   ```bash
   uv run python src/inference_cli.py --list_devices
   ```
2. Run the standalone voice changer:
   ```bash
   uv run python src/inference_cli.py \
     --model distilled_store/spongebob_distilled.onnx \
     --vocos distilled_store/vocos.onnx \
     --input_device <YOUR_MIC_INDEX> \
     --output_device <CABLE_INPUT_INDEX>
   ```
3. Set your input device in Discord, Steam, or active game settings to **CABLE Output** (Windows) or **BlackHole 2ch** (macOS).
4. Disable noise suppression (like Krisp) in Discord, as it may clip or interfere with the converted voice pitch. Speak normally, and the converter will route your distilled voice in real-time!
