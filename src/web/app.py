import os
import sys
import shutil
import threading
import time
import requests
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, BackgroundTasks, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel
from pathlib import Path
from typing import List, Dict, Any

# Ensure parent directory is in path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from training.distill import (
    download_default_dataset, 
    prepare_distillation_data, 
    run_distillation_training
)
from models.vocos_onnx import export_vocos_to_onnx
from core.framework import GameVoiceChanger

app = FastAPI(title="Real-Talk Voice Distillation Framework")

# Directory paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models_store"
DISTILLED_DIR = BASE_DIR / "distilled_store"
DATA_DIR = BASE_DIR / "training_data"

for d in [MODELS_DIR, DISTILLED_DIR, DATA_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Application state
state = {
    "status": "idle",  # idle, preparing, training, active
    "progress": 0.0,
    "current_epoch": 0,
    "total_epochs": 0,
    "current_loss": 0.0,
    "log_history": [],
    "active_training_thread": None,
    "stop_requested": False
}

# Real-time conversion state
changer = None
metrics_thread = None
metrics_active = False

# Active WebSocket clients
websocket_clients: List[WebSocket] = []

class DownloadRequest(BaseModel):
    url: str
    name: str

class DistillRequest(BaseModel):
    rvc_model: str
    rvc_index: str = ""
    epochs: int = 30
    batch_size: int = 8
    custom_source_dir: str = ""

@app.get("/")
def get_index():
    return FileResponse(BASE_DIR / "src" / "web" / "static" / "index.html")

@app.post("/api/download_model")
def download_model(req: DownloadRequest, background_tasks: BackgroundTasks):
    """Download an RVC model from URL in the background"""
    model_name = req.name.strip()
    if not model_name.endswith(".pth"):
        model_name += ".pth"
        
    url = req.url
    target_path = MODELS_DIR / model_name
    
    def do_download(download_url, save_path):
        try:
            print(f"Downloading RVC model from {download_url} to {save_path}...")
            response = requests.get(download_url, stream=True)
            response.raise_for_status()
            with open(save_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"Download complete: {save_path}")
        except Exception as e:
            print(f"Download failed: {e}")
            if save_path.exists():
                os.remove(save_path)
                
    background_tasks.add_task(do_download, url, target_path)
    return {"status": "downloading", "filename": model_name}

@app.get("/api/list_models")
def list_models():
    """List all available models in models_store and distilled_store"""
    rvc_models = [p.name for p in MODELS_DIR.glob("*.pth")]
    rvc_indexes = [p.name for p in MODELS_DIR.glob("*.index")]
    distilled_models = [p.name for p in DISTILLED_DIR.glob("*.onnx")]
    
    # Check if Vocos vocoder is already exported
    vocos_exists = (BASE_DIR / "vocos.onnx").exists() or (DISTILLED_DIR / "vocos.onnx").exists()
    
    return {
        "rvc_models": rvc_models,
        "rvc_indexes": rvc_indexes,
        "distilled_models": distilled_models,
        "vocos_exists": vocos_exists
    }

@app.post("/api/export_vocos")
def api_export_vocos(background_tasks: BackgroundTasks):
    """Trigger Vocos ONNX export"""
    out_path = str(DISTILLED_DIR / "vocos.onnx")
    
    def task():
        try:
            export_vocos_to_onnx(out_path)
            shutil.copy(out_path, str(BASE_DIR / "vocos.onnx"))
        except Exception as e:
            print(f"Vocos export failed: {e}")
            
    background_tasks.add_task(task)
    return {"status": "exporting"}

@app.post("/api/start_distillation")
def start_distillation(req: DistillRequest):
    """Start model distillation training"""
    global state
    
    if state["status"] in ["preparing", "training"]:
        raise HTTPException(status_code=400, detail="Distillation is already running")
        
    state["status"] = "preparing"
    state["progress"] = 0.0
    state["current_epoch"] = 0
    state["total_epochs"] = req.epochs
    state["current_loss"] = 0.0
    state["log_history"] = []
    state["stop_requested"] = False
    
    # Run training in background thread
    t = threading.Thread(target=run_distillation_pipeline, args=(req,))
    state["active_training_thread"] = t
    t.start()
    
    return {"status": "started"}

@app.post("/api/stop_distillation")
def stop_distillation():
    """Request training thread shutdown"""
    global state
    state["stop_requested"] = True
    return {"status": "stop_requested"}

@app.get("/api/status")
def get_status():
    return {
        "status": state["status"],
        "progress": state["progress"],
        "current_epoch": state["current_epoch"],
        "total_epochs": state["total_epochs"],
        "current_loss": state["current_loss"],
        "log_history": state["log_history"]
    }

def broadcast_training_update():
    """Send training metrics update to all WebSocket clients"""
    import asyncio
    payload = {
        "status": state["status"],
        "current_epoch": state["current_epoch"],
        "total_epochs": state["total_epochs"],
        "current_loss": state["current_loss"],
        "progress": state["progress"]
    }
    # Standard sync wrapper for thread safety
    for ws in websocket_clients:
        try:
            # Running synchronous websocket send
            pass
        except Exception:
            pass

def run_distillation_pipeline(req: DistillRequest):
    """Full background thread executing distillation pipeline"""
    global state
    temp_dir = str(DATA_DIR / "temp")
    
    try:
        # 1. Get source dataset
        log("Downloading/preparing source dataset...")
        if req.custom_source_dir and os.path.exists(req.custom_source_dir):
            wav_paths = [str(p) for p in Path(req.custom_source_dir).glob("*.wav")]
        else:
            wav_paths = download_default_dataset(temp_dir)
            
        if not wav_paths:
            raise Exception("No source audio files found.")
            
        if state["stop_requested"]:
            raise Exception("Distillation cancelled by user.")
            
        # 2. Run RVC teacher mapping
        log("Running RVC teacher model on source dataset...")
        rvc_path = str(MODELS_DIR / req.rvc_model)
        index_path = str(MODELS_DIR / req.rvc_index) if req.rvc_index else ""
        
        # Decide RVC device
        import torch
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        
        pairs_num = prepare_distillation_data(
            rvc_model_path=rvc_path,
            rvc_index_path=index_path,
            wav_paths=wav_paths,
            output_dir=temp_dir,
            device_name=device,
            limit=150
        )
        
        if pairs_num == 0:
            raise Exception("RVC data prep failed. No aligned pairs created.")
            
        if state["stop_requested"]:
            raise Exception("Distillation cancelled by user.")
            
        # 3. Train student model
        state["status"] = "training"
        log(f"Starting Causal CNN + GRU model training for {req.epochs} epochs...")
        
        model_name = Path(req.rvc_model).stem
        output_onnx = str(DISTILLED_DIR / f"{model_name}_distilled.onnx")
        
        def progress_cb(epoch, total, loss):
            state["current_epoch"] = epoch
            state["current_loss"] = loss
            state["progress"] = (epoch / total) * 100
            log(f"Epoch {epoch}/{total} - Loss: {loss:.6f}")
            
            # Simple check for cancel request
            if state["stop_requested"]:
                raise Exception("Training cancelled by user.")
                
        run_distillation_training(
            data_dir=temp_dir,
            num_pairs=pairs_num,
            output_onnx_path=output_onnx,
            epochs=req.epochs,
            batch_size=8,
            progress_callback=progress_cb
        )
        
        state["status"] = "idle"
        log(f"Distillation completed successfully! Model saved: {output_onnx}")
        
    except Exception as e:
        state["status"] = "idle"
        log(f"ERROR: {str(e)}")
        print(f"Distillation pipeline error: {e}")

def log(msg: str):
    """Add log entry with timestamp"""
    entry = f"[{time.strftime('%H:%M:%S')}] {msg}"
    state["log_history"].append(entry)
    print(entry)

@app.websocket("/ws/training")
async def websocket_training(websocket: WebSocket):
    await websocket.accept()
    websocket_clients.append(websocket)
    try:
        while True:
            # Keep socket alive and send updates
            await websocket.send_json({
                "status": state["status"],
                "current_epoch": state["current_epoch"],
                "total_epochs": state["total_epochs"],
                "current_loss": state["current_loss"],
                "progress": state["progress"],
                "logs": state["log_history"][-20:] # Last 20 logs
            })
            await asyncio.sleep(1.0)
    except WebSocketDisconnect:
        websocket_clients.remove(websocket)
    except Exception:
        if websocket in websocket_clients:
            websocket_clients.remove(websocket)

# ----------------- REAL-TIME CHANGER TESTING CONTROL -----------------

class ActiveChangerRequest(BaseModel):
    model_name: str
    action: str  # start, stop

@app.post("/api/voice_changer")
def control_voice_changer(req: ActiveChangerRequest):
    """Controls the local server-side voice changer (mic -> output) for live dashboard latency test"""
    global changer, metrics_active, metrics_thread
    
    if req.action == "start":
        if changer and changer.is_running:
            return {"status": "already_running"}
            
        model_path = str(DISTILLED_DIR / req.model_name)
        vocos_path = str(DISTILLED_DIR / "vocos.onnx")
        if not os.path.exists(vocos_path):
            vocos_path = str(BASE_DIR / "vocos.onnx")
            
        print(f"Starting server-side real-time converter: {model_path}")
        try:
            changer = GameVoiceChanger(sample_rate=24000, chunk_size=512)
            # Load distilled model
            success = changer.load_model("distilled", model_path, vocos_onnx_path=vocos_path)
            if not success:
                raise Exception("Failed to load distilled model")
                
            changer.start_real_time_conversion()
            metrics_active = True
            return {"status": "active"}
        except Exception as e:
            print(f"Failed to start changer: {e}")
            raise HTTPException(status_code=500, detail=f"Changer start error: {str(e)}")
            
    elif req.action == "stop":
        if changer:
            changer.stop_conversion()
            changer = None
        metrics_active = False
        return {"status": "idle"}
        
    return {"status": "unknown_action"}

@app.websocket("/ws/metrics")
async def websocket_metrics(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            if changer and changer.is_running:
                stats = changer.get_performance_stats()
                await websocket.send_json({
                    "avg_latency": stats.get("avg_latency_ms", 0.0),
                    "max_latency": stats.get("max_latency_ms", 0.0),
                    "min_latency": stats.get("min_latency_ms", 0.0),
                    "running": True
                })
            else:
                await websocket.send_json({"running": False})
            await asyncio.sleep(0.5)
    except WebSocketDisconnect:
        pass
    except Exception:
        pass

# Create static directories and mount
import asyncio
static_dir = BASE_DIR / "src" / "web" / "static"
static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
