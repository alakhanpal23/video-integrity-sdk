import os
import subprocess
import tempfile
import time
import shutil
import requests
import numpy as np
import pytest
from multiprocessing import Process
import cv2

API_URL = "http://127.0.0.1:8000/verify"
SECRET_HEX = "ffeeddccbbaa99887766554433221100"

# test_embed_verify.py: End-to-end tests for watermark embed/verify
#
# - test_verify_success: Embeds a fixed secret, verifies BER < 0.1
# - test_verify_corrupted: Corrupts a frame, expects BER > 0.2
# - uvicorn_server: Pytest fixture to start/stop API server
# - watermarked_video: Fixture to generate a test video with watermark
#
# To add new test scenarios, define new test_ functions below.

# Helper to generate a synthetic test video using OpenCV
def generate_test_video(path, num_frames=20, size=256, fps=10):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(path, fourcc, fps, (size, size))
    for i in range(num_frames):
        # Color bars pattern
        img = np.zeros((size, size, 3), dtype=np.uint8)
        img[:, :size//3] = (255, 0, 0)
        img[:, size//3:2*size//3] = (0, 255, 0)
        img[:, 2*size//3:] = (0, 0, 255)
        img = np.roll(img, i*5, axis=1)  # animate
        out.write(img)
    out.release()

@pytest.fixture(scope="module")
def uvicorn_server():
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    proc = subprocess.Popen([
        "uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"
    ], env=env)
    time.sleep(2)  # Wait for server to start
    yield
    proc.terminate()
    proc.wait()

@pytest.fixture(scope="module")
def watermarked_video(tmp_path_factory):
    tmpdir = tmp_path_factory.mktemp("embedtest")
    in_mp4 = str(tmpdir / "in.mp4")
    out_mp4 = str(tmpdir / "out.mp4")
    # Generate a simple test video (color bars) using OpenCV
    generate_test_video(in_mp4)
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    subprocess.run([
        "python", "src/encoder.py", "embed", in_mp4, out_mp4,
        "--secret", SECRET_HEX
    ], check=True, env=env)
    return out_mp4

def test_verify_success(uvicorn_server, watermarked_video):
    with open(watermarked_video, "rb") as f:
        files = {"file": ("video.mp4", f, "video/mp4")}
        data = {"secret_hex": SECRET_HEX, "n_frames": 8}
        resp = requests.post(API_URL, files=files, data=data)
    assert resp.status_code == 200
    result = resp.json()
    assert result["ber"] < 0.1
    assert result["valid"] is True

def test_verify_corrupted(uvicorn_server, watermarked_video, tmp_path):
    # Copy and corrupt one frame in the video
    corrupted_mp4 = str(tmp_path / "corrupt.mp4")
    shutil.copy(watermarked_video, corrupted_mp4)
    # Extract frames using OpenCV
    cap = cv2.VideoCapture(corrupted_mp4)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    # Blackout the first frame
    if frames:
        frames[0][:] = 0
    # Re-encode video using OpenCV
    size = frames[0].shape[1], frames[0].shape[0]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    reenc_mp4 = str(tmp_path / "reenc.mp4")
    out = cv2.VideoWriter(reenc_mp4, fourcc, 10, size)
    for frame in frames:
        out.write(frame)
    out.release()
    with open(reenc_mp4, "rb") as f:
        files = {"file": ("video.mp4", f, "video/mp4")}
        data = {"secret_hex": SECRET_HEX, "n_frames": 8}
        resp = requests.post(API_URL, files=files, data=data)
    assert resp.status_code == 200
    result = resp.json()
    assert result["ber"] > 0.2
    assert result["valid"] is False 