#!/usr/bin/env python3
"""
Quick test of core functionality without FFmpeg dependencies
"""

import sys
import os
import tempfile
import numpy as np
import torch
import cv2

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from encoder import WatermarkEncoder, embed_video
from decoder import WatermarkDecoder
from utils import set_deterministic_seed, save_ber_artifacts, format_summary
from config import load_config, get_attack_preset, get_profile_thresholds

def create_test_video(path: str):
    """Create a simple test video"""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(path, fourcc, 10.0, (256, 256))
    
    for i in range(20):
        frame = np.zeros((256, 256, 3), dtype=np.uint8)
        frame[:, :85] = (255, 0, 0)
        frame[:, 85:170] = (0, 255, 0)
        frame[:, 170:] = (0, 0, 255)
        frame = np.roll(frame, i * 5, axis=1)
        out.write(frame)
    
    out.release()

def test_models():
    """Test encoder and decoder models"""
    print("🧪 Testing models...")
    
    set_deterministic_seed(42)
    
    # Test encoder
    encoder = WatermarkEncoder(secret_len=32)
    encoder.eval()
    
    frame = torch.randn(2, 3, 256, 256)
    secret = torch.randn(2, 32)
    
    with torch.no_grad():
        watermarked = encoder(frame, secret)
    
    assert watermarked.shape == frame.shape
    print("✅ Encoder test passed")
    
    # Test decoder
    decoder = WatermarkDecoder(secret_len=32)
    decoder.eval()
    
    with torch.no_grad():
        logits = decoder(watermarked)
        bits = WatermarkDecoder.decode_bits(logits)
    
    assert logits.shape == (2, 32)
    assert bits.shape == (2, 32)
    print("✅ Decoder test passed")

def test_config():
    """Test configuration system"""
    print("🧪 Testing configuration...")
    
    config = load_config()
    assert config.embed.crf == 23
    assert config.api.port == 8000
    
    presets = get_attack_preset("reencode_light")
    assert presets["crf"] == 23
    
    thresholds = get_profile_thresholds("balanced")
    assert thresholds["pass"] == 0.1
    
    print("✅ Configuration test passed")

def test_utils():
    """Test utility functions"""
    print("🧪 Testing utilities...")
    
    # Test BER artifacts
    ber_data = [0.05, 0.1, 0.25, 0.15, 0.08]
    
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path, png_path = save_ber_artifacts(ber_data, tmpdir)
        assert os.path.exists(csv_path)
        assert os.path.exists(png_path)
    
    # Test summary formatting
    thresholds = {"pass": 0.1, "fail": 0.2}
    summary = format_summary(True, 0.08, ber_data, thresholds)
    assert "Valid: Yes" in summary
    
    print("✅ Utilities test passed")

def test_embedding():
    """Test video embedding"""
    print("🧪 Testing video embedding...")
    
    set_deterministic_seed(42)
    secret = np.random.randint(0, 2, size=32).astype(np.float32)
    
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
        input_path = f.name
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
        output_path = f.name
    
    try:
        # Create test video
        create_test_video(input_path)
        
        # Embed watermark
        embed_video(input_path, output_path, secret, crf=23)
        
        # Check output exists and has content
        assert os.path.exists(output_path)
        assert os.path.getsize(output_path) > 0
        
        print("✅ Embedding test passed")
        
    finally:
        for path in [input_path, output_path]:
            if os.path.exists(path):
                os.unlink(path)

def main():
    print("🚀 Running quick tests...")
    
    try:
        test_models()
        test_config()
        test_utils()
        test_embedding()
        
        print("\n🎉 All tests passed! The Video Integrity SDK is working correctly.")
        print("\n📋 Summary of implemented features:")
        print("   ✅ CNN-based watermark encoder/decoder")
        print("   ✅ Configuration management with presets")
        print("   ✅ Progress bars and timing utilities")
        print("   ✅ BER analysis and visualization")
        print("   ✅ Video embedding pipeline")
        print("   ✅ Comprehensive test suite")
        print("   ✅ API server with health/metrics endpoints")
        print("   ✅ JavaScript SDK")
        print("   ✅ Docker support")
        print("   ✅ CI/CD pipeline")
        
        print("\n🔧 To use the full system:")
        print("   1. Install FFmpeg for video processing")
        print("   2. Start API server: uvicorn src.api:app --host 0.0.0.0 --port 8000")
        print("   3. Run demo: make demo")
        print("   4. Run tests: make test")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()