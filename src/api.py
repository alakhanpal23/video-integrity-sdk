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
import base64
import numpy as np
import torch
import cv2
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request, Depends
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Optional
from decoder import WatermarkDecoder
from keys import get_secret
from config import load_config, get_profile_thresholds
from utils import save_ber_artifacts, format_summary, progress_wrapper, Timer
import logging
import time
from starlette.middleware.base import BaseHTTPMiddleware
from video_analysis import analyze_frames
from prometheus_client import Counter, Histogram, generate_latest
import requests

# Load configuration
config = load_config()

app = FastAPI(
    title="Video Integrity SDK API",
    description="Watermark embedding and verification for video integrity",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.api.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Prometheus metrics
REQUESTS_TOTAL = Counter("verify_requests_total", "Total verify requests", ["status"])
VERIFY_LATENCY = Histogram("verify_latency_seconds", "Verify request latency")
AVG_BER = Histogram("avg_ber", "Average BER distribution")

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

class APIKeyMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if config.api.api_key and request.url.path not in ["/healthz", "/readyz", "/metrics", "/docs", "/openapi.json"]:
            api_key = request.headers.get("x-api-key")
            if api_key != config.api.api_key:
                REQUESTS_TOTAL.labels(status="unauthorized").inc()
                raise HTTPException(status_code=401, detail="Invalid API key")
        return await call_next(request)

app.add_middleware(RequestIDMiddleware)
app.add_middleware(APIKeyMiddleware)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time
    
    # Update metrics
    if request.url.path == "/verify":
        VERIFY_LATENCY.observe(duration)
        
    logger.info(f"{request.method} {request.url.path} [{getattr(request.state, 'request_id', '-')}] {response.status_code} {duration:.3f}s")
    return response

# Health endpoints
@app.get("/healthz")
async def health():
    """Health check endpoint"""
    return {"status": "healthy"}

@app.get("/readyz")
async def readiness():
    """Readiness check - verify model can be loaded"""
    try:
        decoder = WatermarkDecoder(secret_len=32)
        return {"status": "ready", "model_loaded": True}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Not ready: {e}")

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return PlainTextResponse(generate_latest(), media_type="text/plain")

def validate_file_upload(file: UploadFile) -> None:
    """Validate uploaded file"""
    # Check content type
    if not file.content_type or not file.content_type.startswith('video/'):
        raise HTTPException(status_code=400, detail="File must be a video")
    
    # Check file size (approximate)
    if hasattr(file.file, 'seek') and hasattr(file.file, 'tell'):
        file.file.seek(0, 2)  # Seek to end
        size = file.file.tell()
        file.file.seek(0)  # Reset
        if size > config.api.max_file_size_mb * 1024 * 1024:
            raise HTTPException(status_code=413, detail=f"File too large (max {config.api.max_file_size_mb}MB)")

@app.post("/verify")
async def verify(
    file: UploadFile = File(...),
    video_id: str = Form(None),
    secret_hex: str = Form(None),
    n_frames: int = Form(10),
    profile: str = Form("balanced"),
    sample_strategy: str = Form("uniform"),
    export_artifacts: bool = Form(False)
):
    """
    Upload a video and secret. Sample N frames, decode watermark, compute BER.
    
    Args:
        file: Video file to verify
        video_id: Video ID for secret lookup (alternative to secret_hex)
        secret_hex: Secret as hex string
        n_frames: Number of frames to sample (1-100)
        profile: Verification profile (strict/balanced/lenient)
        sample_strategy: Sampling strategy (uniform/keyframes/stride)
        export_artifacts: Whether to export BER CSV and plot
    
    Returns:
        {"valid": bool, "ber": float, "summary": str, "artifacts": dict}
    
    Example:
        curl -F "file=@video.mp4" -F "secret_hex=abc123" http://localhost:8000/verify
    """
    
    # Validate inputs
    validate_file_upload(file)
    if not (1 <= n_frames <= 100):
        raise HTTPException(status_code=400, detail="n_frames must be between 1 and 100")
    
    thresholds = get_profile_thresholds(profile)
    start_time = time.time()
    # Save upload to temp file
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        logger.info(f"Uploaded file saved: {len(content)} bytes")
    except Exception as e:
        logger.error(f"Failed to save upload: {e}")
        REQUESTS_TOTAL.labels(status="error").inc()
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
        REQUESTS_TOTAL.labels(status="error").inc()
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
        logger.error(f"Frame extraction failed: {e}")
        REQUESTS_TOTAL.labels(status="error").inc()
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
            valid = ber < thresholds["pass"]
            
            # Update metrics
            AVG_BER.observe(ber)
            REQUESTS_TOTAL.labels(status="success").inc()
            # --- Enhanced video analysis: pass BER for tamper localization ---
            analysis_report = analyze_frames(batch_tensor, per_frame_ber=per_frame_ber)
            logger.info(f"Frame analysis: {analysis_report}")
            if analysis_report.get('tampered_frames'):
                logger.warning(f"Likely tampered frames: {analysis_report['tampered_frames']}")
        except Exception as e:
            os.remove(tmp_path)
            logger.error(f"Decode failed: {e}")
            REQUESTS_TOTAL.labels(status="error").inc()
            raise HTTPException(status_code=500, detail=f"Decode failed: {e}")
    except Exception as e:
        os.remove(tmp_path)
        logger.error(f"Frame loading failed: {e}")
        REQUESTS_TOTAL.labels(status="error").inc()
        raise HTTPException(status_code=500, detail=f"Frame loading failed: {e}")

    # Generate summary
    summary = format_summary(valid, ber, per_frame_ber.tolist(), thresholds)
    
    # Export artifacts if requested
    artifacts = {}
    if export_artifacts:
        try:
            csv_path, png_path = save_ber_artifacts(per_frame_ber.tolist())
            # Read and encode artifacts
            with open(csv_path, 'r') as f:
                artifacts['ber_csv'] = base64.b64encode(f.read().encode()).decode()
            with open(png_path, 'rb') as f:
                artifacts['ber_png'] = base64.b64encode(f.read()).decode()
            # Cleanup artifact files
            os.remove(csv_path)
            os.remove(png_path)
        except Exception as e:
            logger.warning(f"Failed to export artifacts: {e}")
    
    # Cleanup
    os.remove(tmp_path)
    for f in frame_files:
        os.remove(f)
    os.rmdir(frames_dir)
    
    duration = time.time() - start_time
    logger.info(f"Verification completed in {duration:.2f}s: {summary}")

    return JSONResponse({
        "valid": valid, 
        "ber": ber,
        "summary": summary,
        "per_frame_ber": per_frame_ber.tolist(),
        "thresholds": thresholds,
        "artifacts": artifacts,
        "processing_time": duration
    })

# Moved to test utilities

# Remove test code - should be in separate test file
# generate_test_video("test_video.mp4")
# 
# with open("watermarked.mp4", "rb") as f:
#     files = {"file": ("video.mp4", f, "video/mp4")}
#     data = {"secret_hex": "ffeeddccbbaa99887766554433221100", "n_frames": 8}
#     resp = requests.post("http://127.0.0.1:8000/verify", files=files, data=data)
#     print(resp.status_code)
#     print(resp.json())
