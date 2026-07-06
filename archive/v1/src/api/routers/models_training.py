"""
Models and Training mock router for WiFi-DensePose API.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from fastapi import APIRouter, HTTPException, status, WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)

router = APIRouter()

# Mock storage
active_model_id = "densepose_resnet50_v1"
is_training = False
is_recording = False

MODELS_LIST = [
    {
        "model_id": "densepose_resnet50_v1",
        "name": "DensePose ResNet-50 v1",
        "description": "Baseline CMU model for multi-person tracking",
        "created_at": "2025-01-01T00:00:00Z",
        "accuracy": 0.872,
        "parameters": "25.6M",
        "is_active": True
    },
    {
        "model_id": "densepose_vit_v2",
        "name": "DensePose ViT v2",
        "description": "High-accuracy vision transformer based model",
        "created_at": "2025-06-01T00:00:00Z",
        "accuracy": 0.915,
        "parameters": "86.4M",
        "is_active": False
    }
]

RECORDINGS_LIST = [
    {
        "id": "rec_20250706_120000",
        "name": "Mock CSI Recording 1",
        "timestamp": "2025-07-06T12:00:00Z",
        "duration_seconds": 60.0,
        "file_size_bytes": 1048576,
        "samples_count": 600,
        "status": "completed"
    }
]

# --- Model Endpoints ---

@router.get("/models")
async def list_models():
    """List all available models."""
    global active_model_id
    for model in MODELS_LIST:
        model["is_active"] = (model["model_id"] == active_model_id)
    return {"models": MODELS_LIST}

@router.get("/models/active")
async def get_active_model():
    """Get active model info."""
    global active_model_id
    if not active_model_id:
        return None
    for model in MODELS_LIST:
        if model["model_id"] == active_model_id:
            return model
    return None

@router.get("/models/{model_id}")
async def get_model(model_id: str):
    """Get specific model info."""
    for model in MODELS_LIST:
        if model["model_id"] == model_id:
            return model
    raise HTTPException(status_code=404, detail="Model not found")

@router.post("/models/load")
async def load_model(body: Dict[str, Any]):
    """Load a model."""
    global active_model_id
    model_id = body.get("model_id")
    if not model_id:
        raise HTTPException(status_code=400, detail="Missing model_id")
    found = False
    for model in MODELS_LIST:
        if model["model_id"] == model_id:
            found = True
            break
    if not found:
        raise HTTPException(status_code=404, detail="Model not found")
    active_model_id = model_id
    logger.info(f"Loaded model {model_id}")
    return {"status": "success", "message": f"Model {model_id} loaded successfully"}

@router.post("/models/unload")
async def unload_model():
    """Unload active model."""
    global active_model_id
    active_model_id = None
    logger.info("Unloaded active model")
    return {"status": "success", "message": "Model unloaded successfully"}

@router.get("/models/lora/profiles")
async def list_lora_profiles():
    """List LoRA profiles."""
    return {"profiles": []}

@router.post("/models/lora/activate")
async def activate_lora_profile(body: Dict[str, Any]):
    """Activate LoRA profile."""
    return {"status": "success", "message": "LoRA profile activated successfully"}

@router.delete("/models/{model_id}")
async def delete_model(model_id: str):
    """Delete a model."""
    global active_model_id
    if model_id == active_model_id:
        active_model_id = None
    for idx, model in enumerate(MODELS_LIST):
        if model["model_id"] == model_id:
            MODELS_LIST.pop(idx)
            return {"status": "success", "message": f"Model {model_id} deleted"}
    raise HTTPException(status_code=404, detail="Model not found")


# --- Training Endpoints ---

@router.post("/train/start")
async def start_training(config: Dict[str, Any]):
    """Start training."""
    global is_training
    is_training = True
    logger.info(f"Started training with config: {config}")
    return {"status": "success", "message": "Training started"}

@router.post("/train/stop")
async def stop_training():
    """Stop training."""
    global is_training
    is_training = False
    logger.info("Stopped training")
    return {"status": "success", "message": "Training stopped"}

@router.get("/train/status")
async def get_training_status():
    """Get training status."""
    global is_training
    return {
        "status": "training" if is_training else "idle",
        "current_epoch": 12 if is_training else 0,
        "total_epochs": 100,
        "loss": 0.245 if is_training else 0.0,
        "val_loss": 0.284 if is_training else 0.0,
        "progress": 0.12 if is_training else 0.0,
        "eta_seconds": 3600 if is_training else 0
    }

@router.post("/train/pretrain")
async def start_pretraining(config: Dict[str, Any]):
    """Start pretraining."""
    global is_training
    is_training = True
    logger.info(f"Started pretraining with config: {config}")
    return {"status": "success", "message": "Pretraining started"}

@router.post("/train/lora")
async def start_lora_training(config: Dict[str, Any]):
    """Start LoRA training."""
    global is_training
    is_training = True
    logger.info(f"Started LoRA training with config: {config}")
    return {"status": "success", "message": "LoRA training started"}


# --- Recording Endpoints ---

@router.get("/recording/list")
async def list_recordings():
    """List recordings."""
    return {"recordings": RECORDINGS_LIST}

@router.post("/recording/start")
async def start_recording(config: Dict[str, Any]):
    """Start recording."""
    global is_recording
    is_recording = True
    logger.info(f"Started recording with config: {config}")
    return {"status": "success", "message": "Recording started"}

@router.post("/recording/stop")
async def stop_recording():
    """Stop recording."""
    global is_recording
    is_recording = False
    logger.info("Stopped recording")
    # Add new mock recording to list
    new_rec_id = f"rec_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    RECORDINGS_LIST.append({
        "id": new_rec_id,
        "name": f"Mock CSI Recording {len(RECORDINGS_LIST) + 1}",
        "timestamp": datetime.now().isoformat(),
        "duration_seconds": 60.0,
        "file_size_bytes": 1048576,
        "samples_count": 600,
        "status": "completed"
    })
    return {"status": "success", "message": "Recording stopped", "recording_id": new_rec_id}

@router.delete("/recording/{recording_id}")
async def delete_recording(recording_id: str):
    """Delete a recording."""
    for idx, rec in enumerate(RECORDINGS_LIST):
        if rec["id"] == recording_id:
            RECORDINGS_LIST.pop(idx)
            return {"status": "success", "message": f"Recording {recording_id} deleted"}
    raise HTTPException(status_code=404, detail="Recording not found")


@router.websocket("/ws/train/progress")
async def websocket_train_progress(websocket: WebSocket):
    """WebSocket endpoint for mock training progress."""
    await websocket.accept()
    try:
        import asyncio
        epoch = 1
        while True:
            if is_training:
                await websocket.send_json({
                    "epoch": epoch,
                    "loss": 0.35 - (epoch * 0.002),
                    "val_loss": 0.38 - (epoch * 0.0018),
                    "accuracy": 0.75 + (epoch * 0.002),
                    "progress": epoch / 100.0,
                    "eta_seconds": (100 - epoch) * 30
                })
                epoch = min(epoch + 1, 100)
            else:
                await websocket.send_json({
                    "status": "idle"
                })
            await asyncio.sleep(2.0)
    except WebSocketDisconnect:
        logger.info("Training progress WebSocket disconnected")
    except Exception as e:
        logger.error(f"Error in training progress WebSocket: {e}")

