# api.py: FastAPI server for watermark verification
#
# - /verify: POST endpoint to check watermark integrity in uploaded videos
#   * Accepts video, video_id or secret, samples frames, decodes watermark, returns BER
# - Integrates with keys.py for secret lookup
# - Adds logging, request-ID middleware, and metrics
#
# To add new endpoints, define new @app.post or @app.get routes.
# To change verification logic, edit verify().
# To add more logging/metrics, extend middleware.

import os
import uuid
import tempfile
import subprocess
import numpy as np
import torch
import cv2
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.responses import JSONResponse
from typing import Dict
from decoder import WatermarkDecoder
from keys import get_secret
import logging
import time
from starlette.middleware.base import BaseHTTPMiddleware
from video_analysis import analyze_frames
import requests

app = FastAPI()

# In-memory secret store: {upload_id: secret}
SECRET_STORE: Dict[str, np.ndarray] = {}

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("video_sdk")

class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response

app.add_middleware(RequestIDMiddleware)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time
    logger.info(f"{request.method} {request.url.path} [{getattr(request.state, 'request_id', '-')}] {response.status_code} {duration:.3f}s")
    return response

@app.post("/verify")
async def verify(
    file: UploadFile = File(...),
    video_id: str = Form(None),
    secret_hex: str = Form(None),
    n_frames: int = Form(10)
):
    """
    Upload a video and secret. Sample N frames, decode watermark, compute BER.
    Returns: {"valid": bool, "ber": float}
    """
    # Save upload to temp file
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name
    except Exception as e:
        logger.error(f"Failed to save upload: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save upload: {e}")

    # Get secret
    try:
        if secret_hex:
            secret_bytes = bytes.fromhex(secret_hex)
        elif video_id:
            secret_hex = get_secret(video_id)
            if not secret_hex:
                raise Exception("No secret found for video_id")
            secret_bytes = bytes.fromhex(secret_hex)
        else:
            raise Exception("No secret or video_id provided")
        secret = np.unpackbits(np.frombuffer(secret_bytes, dtype=np.uint8)).astype(np.float32)
        if secret.shape[0] > 32:
            secret = secret[:32]
        elif secret.shape[0] < 32:
            secret = np.pad(secret, (0, 32 - secret.shape[0]), 'constant')
    except Exception as e:
        os.remove(tmp_path)
        logger.error(f"Secret error: {e}")
        raise HTTPException(status_code=400, detail=f"Secret error: {e}")

    # Sample N evenly spaced frames using ffmpeg
    try:
        # Get total frame count
        probe = subprocess.run([
            'ffprobe', '-v', 'error', '-select_streams', 'v:0',
            '-count_frames', '-show_entries', 'stream=nb_read_frames',
            '-of', 'default=nokey=1:noprint_wrappers=1', tmp_path
        ], capture_output=True, text=True)
        total_frames = int(probe.stdout.strip() or 0)
        if total_frames == 0:
            # fallback: try to get nb_frames
            probe2 = subprocess.run([
                'ffprobe', '-v', 'error', '-select_streams', 'v:0',
                '-show_entries', 'stream=nb_frames',
                '-of', 'default=nokey=1:noprint_wrappers=1', tmp_path
            ], capture_output=True, text=True)
            total_frames = int(probe2.stdout.strip() or 0)
        if total_frames == 0:
            os.remove(tmp_path)
            raise HTTPException(status_code=400, detail="Could not determine frame count.")
        step = max(1, total_frames // n_frames)
        # Extract frames
        frames_dir = tempfile.mkdtemp()
        out_pattern = os.path.join(frames_dir, 'frame_%04d.png')
        ffmpeg_cmd = [
            'ffmpeg', '-i', tmp_path,
            '-vf', f'select=not(mod(n\,{step}))',
            '-vsync', 'vfr', '-frames:v', str(n_frames), out_pattern,
            '-hide_banner', '-loglevel', 'error'
        ]
        subprocess.run(ffmpeg_cmd, check=True)
        # Collect frame paths
        frame_files = sorted([os.path.join(frames_dir, f) for f in os.listdir(frames_dir) if f.endswith('.png')])
        if len(frame_files) == 0:
            raise Exception("No frames extracted.")
    except Exception as e:
        os.remove(tmp_path)
        raise HTTPException(status_code=500, detail=f"Frame extraction failed: {e}")

    # Load frames, preprocess, and stack
    try:
        batch = []
        for fpath in frame_files:
            img = cv2.imread(fpath, cv2.IMREAD_COLOR)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (256, 256))
            img = img.astype(np.float32) / 255.0
            img = np.transpose(img, (2,0,1))  # [3,256,256]
            batch.append(img)
        if len(batch) == 0:
            raise Exception("No valid frames loaded.")
        batch_tensor = torch.from_numpy(np.stack(batch)).float()
        # Decode watermark
        try:
            decoder = WatermarkDecoder(secret_len=secret.shape[0])
            decoder.eval()
            with torch.no_grad():
                logits = decoder(batch_tensor)
            bits = WatermarkDecoder.decode_bits(logits)  # [N, secret_len]
            # Per-frame BER
            per_frame_ber = np.mean(bits != secret, axis=1)  # [N]
            logger.info(f'Per-frame BER: {per_frame_ber.tolist()}')
            # Majority vote across frames
            rec_bits = (np.sum(bits, axis=0) > (len(bits)//2)).astype(np.uint8)
            ber = float(np.mean(rec_bits != secret))
            valid = ber < 0.1
            # --- Enhanced video analysis: pass BER for tamper localization ---
            analysis_report = analyze_frames(batch_tensor, per_frame_ber=per_frame_ber)
            logger.info(f"Frame analysis: {analysis_report}")
            if analysis_report.get('tampered_frames'):
                logger.warning(f"Likely tampered frames: {analysis_report['tampered_frames']}")
        except Exception as e:
            os.remove(tmp_path)
            raise HTTPException(status_code=500, detail=f"Decode failed: {e}")
    except Exception as e:
        os.remove(tmp_path)
        raise HTTPException(status_code=500, detail=f"Frame loading failed: {e}")

    # Cleanup
    os.remove(tmp_path)
    for f in frame_files:
        os.remove(f)
    os.rmdir(frames_dir)

    return JSONResponse({"valid": valid, "ber": ber})

def generate_test_video(path, num_frames=20, size=256, fps=10):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(path, fourcc, fps, (size, size))
    for i in range(num_frames):
        img = np.zeros((size, size, 3), dtype=np.uint8)
        img[:, :size//3] = (255, 0, 0)
        img[:, size//3:2*size//3] = (0, 255, 0)
        img[:, 2*size//3:] = (0, 0, 255)
        img = np.roll(img, i*5, axis=1)
        out.write(img)
    out.release()

generate_test_video("test_video.mp4")

with open("watermarked.mp4", "rb") as f:
    files = {"file": ("video.mp4", f, "video/mp4")}
    data = {"secret_hex": "ffeeddccbbaa99887766554433221100", "n_frames": 8}
    resp = requests.post("http://127.0.0.1:8000/verify", files=files, data=data)
    print(resp.status_code)
    print(resp.json())
