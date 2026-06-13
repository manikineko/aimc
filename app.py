"""
FastAPI backend for the interactive Oasis world model.
"""

import io
import os
import base64
import json
import queue
import threading
import time
from typing import Optional

import torch
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from interactive import InteractiveWorld
from utils import ACTION_KEYS, IMAGE_EXTENSIONS, VIDEO_EXTENSIONS

app = FastAPI(title="Oasis Interactive World")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global world instance (loaded lazily)
world: Optional[InteractiveWorld] = None
_world_init_lock = threading.Lock()


def get_world():
    global world
    if world is None:
        with _world_init_lock:
            if world is None:
                oasis_ckpt = os.environ.get("OASIS_CKPT", "oasis500m.safetensors")
                vae_ckpt = os.environ.get("VAE_CKPT", "vit-l-20.safetensors")
                world = InteractiveWorld(
                    oasis_ckpt=oasis_ckpt,
                    vae_ckpt=vae_ckpt,
                    ddim_steps=int(os.environ.get("DDIM_STEPS", "3")),
                    compile=os.environ.get("COMPILE", "0") == "1",
                )
    return world


_step_lock = threading.Lock()
_buffer_queue: queue.Queue = queue.Queue(maxsize=1)
_latest_action = torch.zeros(len(ACTION_KEYS), dtype=torch.float32)
_buffer_stop = threading.Event()


def _buffer_worker():
    """Background thread that pre-generates frames using the latest action."""
    while not _buffer_stop.is_set():
        w = get_world()
        if not w.initialized:
            time.sleep(0.05)
            continue
        if _buffer_queue.full():
            time.sleep(0.02)
            continue
        action = _latest_action.clone()
        with _step_lock:
            try:
                frame = w.step(action)
            except Exception as e:
                print(f"[buffer] step error: {e}")
                time.sleep(0.05)
                continue
        _buffer_queue.put(frame, block=True)


_buffer_thread = threading.Thread(target=_buffer_worker, daemon=True)
_buffer_thread.start()


def actions_dict_to_tensor(actions_dict: dict) -> torch.Tensor:
    """Convert frontend action dict to 25-dim one-hot tensor."""
    actions = torch.zeros(len(ACTION_KEYS), dtype=torch.float32)
    for i, key in enumerate(ACTION_KEYS):
        if key.startswith("camera"):
            # frontend sends normalized values in [-1, 1]
            actions[i] = float(actions_dict.get(key, 0.0))
        else:
            actions[i] = float(actions_dict.get(key, 0.0))
    return actions


def frame_to_bytes(frame_np: np.ndarray, fmt="JPEG") -> bytes:
    img = Image.fromarray(frame_np)
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


@app.post("/api/init")
async def init_world(
    prompt: UploadFile = File(...),
    n_prompt_frames: int = Form(1),
):
    """Upload an image/video to initialize the world. Returns the first frame."""
    contents = await prompt.read()
    ext = os.path.splitext(prompt.filename or "")[1].lower()

    # Detect extension from content if missing / unrecognized
    if not ext or ext[1:] not in (IMAGE_EXTENSIONS | VIDEO_EXTENSIONS):
        if contents[:8] == b"\x89PNG\x0d\x0a\x1a\x0a":
            ext = ".png"
        elif contents[:3] == b"\xff\xd8\xff":
            ext = ".jpg"
        elif contents[:4] == b"\x00\x00\x00 ftyp":
            ext = ".mp4"
        else:
            ext = ".png"

    # Save uploaded file temporarily with a unique name
    tmp_path = f"/tmp/oasis_prompt_{os.urandom(4).hex()}{ext}"
    with open(tmp_path, "wb") as f:
        f.write(contents)

    w = get_world()
    # Flush any stale buffered frames before re-initializing
    while not _buffer_queue.empty():
        try:
            _buffer_queue.get_nowait()
        except queue.Empty:
            break
    frame = w.initialize(tmp_path, n_prompt_frames=n_prompt_frames)

    # Clean up temp file
    os.remove(tmp_path)

    img_bytes = frame_to_bytes(frame)
    return JSONResponse({
        "status": "ok",
        "frame_count": w.frame_count,
        "frame_b64": base64.b64encode(img_bytes).decode("utf-8"),
    })


@app.post("/api/init_default")
async def init_world_default(
    prompt_path: str = Form("sample_data/sample_image_0.png"),
    n_prompt_frames: int = Form(1),
):
    """Initialize world with a file path on the server."""
    w = get_world()
    while not _buffer_queue.empty():
        try:
            _buffer_queue.get_nowait()
        except queue.Empty:
            break
    frame = w.initialize(prompt_path, n_prompt_frames=n_prompt_frames)
    img_bytes = frame_to_bytes(frame)
    return JSONResponse({
        "status": "ok",
        "frame_count": w.frame_count,
        "frame_b64": base64.b64encode(img_bytes).decode("utf-8"),
    })


@app.post("/api/step")
async def step(actions_json: str = Form(...)):
    """Take an action and generate the next frame.
    Returns a pre-buffered frame if available, otherwise generates synchronously."""
    actions_dict = json.loads(actions_json)
    action_tensor = actions_dict_to_tensor(actions_dict)

    w = get_world()
    # update desired action for the background worker
    with _step_lock:
        _latest_action[:] = action_tensor

    # try lock-free pop first — if it succeeds the worker can keep generating
    try:
        frame = _buffer_queue.get_nowait()
    except queue.Empty:
        with _step_lock:
            frame = w.step(action_tensor)
    img_bytes = frame_to_bytes(frame)
    return JSONResponse({
        "status": "ok",
        "frame_count": w.frame_count,
        "frame_b64": base64.b64encode(img_bytes).decode("utf-8"),
    })


@app.post("/api/reset")
async def reset():
    """Reset the world state."""
    w = get_world()
    while not _buffer_queue.empty():
        try:
            _buffer_queue.get_nowait()
        except queue.Empty:
            break
    w.reset_state()
    return JSONResponse({"status": "reset"})


# Serve static frontend
if os.path.exists("static"):
    app.mount("/", StaticFiles(directory="static", html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
