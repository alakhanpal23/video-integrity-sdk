# tests/test_api.py

import pytest
import tempfile
import os
import io
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import cv2
import numpy as np

# Mock the heavy dependencies for faster testing
@pytest.fixture(autouse=True)
def mock_heavy_imports():
    with patch('src.api.WatermarkDecoder') as mock_decoder, \
         patch('src.api.analyze_frames') as mock_analyze, \
         patch('src.api.save_ber_artifacts') as mock_artifacts:
        
        # Mock decoder
        mock_decoder_instance = MagicMock()
        mock_decoder_instance.eval.return_value = None
        mock_decoder.return_value = mock_decoder_instance
        mock_decoder.decode_bits.return_value = np.array([[0, 1, 0, 1] * 8])  # 32 bits
        
        # Mock analysis
        mock_analyze.return_value = {
            'num_frames': 5,
            'warnings': [],
            'tampered_frames': []
        }
        
        # Mock artifacts
        mock_artifacts.return_value = ('/tmp/ber.csv', '/tmp/ber.png')
        
        yield {
            'decoder': mock_decoder,
            'analyze': mock_analyze,
            'artifacts': mock_artifacts
        }

@pytest.fixture
def client():
    """Create test client"""
    # Set test environment
    os.environ['API_KEY'] = 'test-api-key'
    
    from src.api import app
    return TestClient(app)

@pytest.fixture
def test_video_file():
    """Create a small test video file"""
    # Create a simple test video
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
        # Write minimal MP4 header (fake but sufficient for testing)
        f.write(b'\x00\x00\x00\x20ftypmp4\x00\x00\x00\x00mp41isom')
        f.write(b'\x00' * 100)  # Padding
        temp_path = f.name
    
    yield temp_path
    
    if os.path.exists(temp_path):
        os.unlink(temp_path)

def test_health_endpoint(client):
    """Test health check endpoint"""
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_readiness_endpoint(client):
    """Test readiness check endpoint"""
    response = client.get("/readyz")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ready"
    assert data["model_loaded"] is True

def test_metrics_endpoint(client):
    """Test Prometheus metrics endpoint"""
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]

def test_api_key_required(client):
    """Test that API key is required for protected endpoints"""
    # Request without API key
    response = client.post("/verify")
    assert response.status_code == 401
    assert "Invalid API key" in response.json()["detail"]

def test_api_key_valid(client, test_video_file):
    """Test that valid API key allows access"""
    with patch('subprocess.run') as mock_subprocess, \
         patch('cv2.imread') as mock_imread, \
         patch('torch.no_grad'), \
         patch('os.listdir') as mock_listdir:
        
        # Mock subprocess calls
        mock_subprocess.return_value.stdout = "10"
        mock_subprocess.return_value.returncode = 0
        
        # Mock frame loading
        mock_listdir.return_value = ['frame_0001.png']
        mock_imread.return_value = np.ones((256, 256, 3), dtype=np.uint8) * 128
        
        with open(test_video_file, 'rb') as f:
            response = client.post(
                "/verify",
                files={"file": ("test.mp4", f, "video/mp4")},
                data={"secret_hex": "deadbeef", "n_frames": "5"},
                headers={"x-api-key": "test-api-key"}
            )
        
        assert response.status_code == 200

def test_verify_missing_secret(client, test_video_file):
    """Test verification without secret"""
    with open(test_video_file, 'rb') as f:
        response = client.post(
            "/verify",
            files={"file": ("test.mp4", f, "video/mp4")},
            data={"n_frames": "5"},
            headers={"x-api-key": "test-api-key"}
        )
    
    assert response.status_code == 400
    assert "Secret error" in response.json()["detail"]

def test_verify_invalid_file_type(client):
    """Test verification with non-video file"""
    fake_file = io.BytesIO(b"not a video")
    
    response = client.post(
        "/verify",
        files={"file": ("test.txt", fake_file, "text/plain")},
        data={"secret_hex": "deadbeef"},
        headers={"x-api-key": "test-api-key"}
    )
    
    assert response.status_code == 400
    assert "File must be a video" in response.json()["detail"]

def test_verify_file_too_large(client):
    """Test verification with oversized file"""
    # Create a large fake video file
    large_content = b'\x00\x00\x00\x20ftypmp4\x00\x00\x00\x00mp41isom' + b'\x00' * (200 * 1024 * 1024)
    fake_file = io.BytesIO(large_content)
    
    response = client.post(
        "/verify",
        files={"file": ("large.mp4", fake_file, "video/mp4")},
        data={"secret_hex": "deadbeef"},
        headers={"x-api-key": "test-api-key"}
    )
    
    assert response.status_code == 413
    assert "File too large" in response.json()["detail"]

def test_verify_invalid_n_frames(client, test_video_file):
    """Test verification with invalid n_frames parameter"""
    with open(test_video_file, 'rb') as f:
        response = client.post(
            "/verify",
            files={"file": ("test.mp4", f, "video/mp4")},
            data={"secret_hex": "deadbeef", "n_frames": "200"},  # Too many
            headers={"x-api-key": "test-api-key"}
        )
    
    assert response.status_code == 400
    assert "n_frames must be between 1 and 100" in response.json()["detail"]

@patch('subprocess.run')
@patch('cv2.imread')
@patch('torch.no_grad')
@patch('os.listdir')
def test_verify_success(mock_listdir, mock_no_grad, mock_imread, mock_subprocess, client, test_video_file):
    """Test successful verification"""
    # Mock subprocess calls
    mock_subprocess.return_value.stdout = "10"
    mock_subprocess.return_value.returncode = 0
    
    # Mock frame loading
    mock_listdir.return_value = ['frame_0001.png', 'frame_0002.png']
    mock_imread.return_value = np.ones((256, 256, 3), dtype=np.uint8) * 128
    
    with open(test_video_file, 'rb') as f:
        response = client.post(
            "/verify",
            files={"file": ("test.mp4", f, "video/mp4")},
            data={
                "secret_hex": "deadbeef",
                "n_frames": "5",
                "profile": "balanced",
                "export_artifacts": "true"
            },
            headers={"x-api-key": "test-api-key"}
        )
    
    assert response.status_code == 200
    data = response.json()
    
    # Check response structure
    assert "valid" in data
    assert "ber" in data
    assert "summary" in data
    assert "per_frame_ber" in data
    assert "thresholds" in data
    assert "processing_time" in data
    assert isinstance(data["valid"], bool)
    assert isinstance(data["ber"], float)

def test_verify_with_video_id(client, test_video_file):
    """Test verification using video_id instead of secret_hex"""
    with patch('src.api.get_secret') as mock_get_secret, \
         patch('subprocess.run') as mock_subprocess, \
         patch('cv2.imread') as mock_imread, \
         patch('torch.no_grad'), \
         patch('os.listdir') as mock_listdir:
        
        # Mock secret lookup
        mock_get_secret.return_value = "deadbeef"
        
        # Mock subprocess and frame loading
        mock_subprocess.return_value.stdout = "10"
        mock_listdir.return_value = ['frame_0001.png']
        mock_imread.return_value = np.ones((256, 256, 3), dtype=np.uint8) * 128
        
        with open(test_video_file, 'rb') as f:
            response = client.post(
                "/verify",
                files={"file": ("test.mp4", f, "video/mp4")},
                data={"video_id": "test_video_123", "n_frames": "5"},
                headers={"x-api-key": "test-api-key"}
            )
        
        assert response.status_code == 200
        mock_get_secret.assert_called_once_with("test_video_123")

def test_verify_video_id_not_found(client, test_video_file):
    """Test verification with non-existent video_id"""
    with patch('src.api.get_secret') as mock_get_secret:
        mock_get_secret.return_value = None
        
        with open(test_video_file, 'rb') as f:
            response = client.post(
                "/verify",
                files={"file": ("test.mp4", f, "video/mp4")},
                data={"video_id": "nonexistent"},
                headers={"x-api-key": "test-api-key"}
            )
        
        assert response.status_code == 400
        assert "No secret found for video_id" in response.json()["detail"]

def test_cors_headers(client):
    """Test CORS headers are present"""
    response = client.options("/verify", headers={"Origin": "http://localhost:3000"})
    # FastAPI handles OPTIONS automatically with CORS middleware
    assert response.status_code in [200, 405]  # 405 if OPTIONS not explicitly defined

def test_request_id_header(client):
    """Test that request ID is added to response headers"""
    response = client.get("/healthz")
    assert "X-Request-ID" in response.headers
    assert len(response.headers["X-Request-ID"]) > 0

@patch('subprocess.run')
def test_frame_extraction_failure(mock_subprocess, client, test_video_file):
    """Test handling of frame extraction failure"""
    # Mock ffprobe to return 0 frames
    mock_subprocess.return_value.stdout = "0"
    mock_subprocess.return_value.returncode = 0
    
    with open(test_video_file, 'rb') as f:
        response = client.post(
            "/verify",
            files={"file": ("test.mp4", f, "video/mp4")},
            data={"secret_hex": "deadbeef"},
            headers={"x-api-key": "test-api-key"}
        )
    
    assert response.status_code == 400
    assert "Could not determine frame count" in response.json()["detail"]

def test_different_profiles(client, test_video_file):
    """Test different verification profiles"""
    profiles = ["strict", "balanced", "lenient"]
    
    for profile in profiles:
        with patch('subprocess.run') as mock_subprocess, \
             patch('cv2.imread') as mock_imread, \
             patch('torch.no_grad'), \
             patch('os.listdir') as mock_listdir:
            
            mock_subprocess.return_value.stdout = "10"
            mock_listdir.return_value = ['frame_0001.png']
            mock_imread.return_value = np.ones((256, 256, 3), dtype=np.uint8) * 128
            
            with open(test_video_file, 'rb') as f:
                response = client.post(
                    "/verify",
                    files={"file": ("test.mp4", f, "video/mp4")},
                    data={"secret_hex": "deadbeef", "profile": profile},
                    headers={"x-api-key": "test-api-key"}
                )
            
            assert response.status_code == 200
            data = response.json()
            assert "thresholds" in data
            assert "pass" in data["thresholds"]
            assert "fail" in data["thresholds"]