# tests/test_integration.py

import pytest
import tempfile
import os
import subprocess
import numpy as np
import cv2
from src.encoder import embed_video
from src.cli import verify_video
from src.utils import set_deterministic_seed

@pytest.fixture
def test_video():
    """Create a simple test video"""
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
        temp_path = f.name
    
    # Create a simple test video using OpenCV
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_path, fourcc, 10.0, (256, 256))
    
    for i in range(20):  # 20 frames
        # Create a simple pattern
        frame = np.zeros((256, 256, 3), dtype=np.uint8)
        frame[:, :85] = (255, 0, 0)      # Red
        frame[:, 85:170] = (0, 255, 0)   # Green  
        frame[:, 170:] = (0, 0, 255)     # Blue
        
        # Add some motion
        frame = np.roll(frame, i * 5, axis=1)
        out.write(frame)
    
    out.release()
    
    yield temp_path
    
    if os.path.exists(temp_path):
        os.unlink(temp_path)

def test_embed_and_verify_integration(test_video):
    """Test full embed and verify pipeline"""
    set_deterministic_seed(42)
    
    # Generate a test secret
    secret = np.random.randint(0, 2, size=32).astype(np.float32)
    
    # Create output path
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
        watermarked_path = f.name
    
    try:
        # Embed watermark
        embed_video(test_video, watermarked_path, secret, crf=23)
        
        # Verify the file was created
        assert os.path.exists(watermarked_path)
        assert os.path.getsize(watermarked_path) > 0
        
        # Verify watermark
        result = verify_video(
            watermarked_path, 
            secret, 
            n_frames=5,
            profile="balanced"
        )
        
        # Check results
        assert isinstance(result, dict)
        assert "valid" in result
        assert "ber" in result
        assert "summary" in result
        
        # BER should be low for a freshly watermarked video
        assert result["ber"] < 0.2, f"BER too high: {result['ber']}"
        
    finally:
        if os.path.exists(watermarked_path):
            os.unlink(watermarked_path)

def test_different_secrets_produce_different_results(test_video):
    """Test that different secrets produce different verification results"""
    set_deterministic_seed(42)
    
    # Two different secrets
    secret1 = np.random.randint(0, 2, size=32).astype(np.float32)
    set_deterministic_seed(123)
    secret2 = np.random.randint(0, 2, size=32).astype(np.float32)
    
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
        watermarked_path = f.name
    
    try:
        # Embed with secret1
        embed_video(test_video, watermarked_path, secret1, crf=23)
        
        # Verify with secret1 (should work)
        result1 = verify_video(watermarked_path, secret1, n_frames=5)
        
        # Verify with secret2 (should fail)
        result2 = verify_video(watermarked_path, secret2, n_frames=5)
        
        # Results should be different
        assert result1["ber"] != result2["ber"]
        # Correct secret should have lower BER
        assert result1["ber"] < result2["ber"]
        
    finally:
        if os.path.exists(watermarked_path):
            os.unlink(watermarked_path)

def test_compression_robustness(test_video):
    """Test watermark survives compression"""
    set_deterministic_seed(42)
    secret = np.random.randint(0, 2, size=32).astype(np.float32)
    
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
        watermarked_path = f.name
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
        compressed_path = f.name
    
    try:
        # Embed watermark
        embed_video(test_video, watermarked_path, secret, crf=18)
        
        # Compress the video (simulate attack)
        subprocess.run([
            'ffmpeg', '-i', watermarked_path, '-c:v', 'libx264', 
            '-crf', '28', compressed_path, '-y', '-hide_banner', '-loglevel', 'error'
        ], check=True)
        
        # Verify both versions
        result_original = verify_video(watermarked_path, secret, n_frames=5)
        result_compressed = verify_video(compressed_path, secret, n_frames=5)
        
        # Both should be valid, but compressed version may have higher BER
        assert result_original["valid"]
        assert result_compressed["ber"] >= result_original["ber"]
        
        # Compressed version should still be reasonably good
        assert result_compressed["ber"] < 0.3, f"Compression destroyed watermark: BER={result_compressed['ber']}"
        
    finally:
        for path in [watermarked_path, compressed_path]:
            if os.path.exists(path):
                os.unlink(path)

def test_frame_sampling_strategies(test_video):
    """Test different frame sampling strategies"""
    set_deterministic_seed(42)
    secret = np.random.randint(0, 2, size=32).astype(np.float32)
    
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
        watermarked_path = f.name
    
    try:
        embed_video(test_video, watermarked_path, secret, crf=23)
        
        strategies = ["uniform", "keyframes"]
        results = {}
        
        for strategy in strategies:
            result = verify_video(
                watermarked_path, 
                secret, 
                n_frames=5,
                sample_strategy=strategy
            )
            results[strategy] = result
            
            # All strategies should work
            assert result["ber"] < 0.3, f"Strategy {strategy} failed: BER={result['ber']}"
        
        # Results may differ slightly due to different sampling
        print(f"Uniform BER: {results['uniform']['ber']:.4f}")
        print(f"Keyframes BER: {results['keyframes']['ber']:.4f}")
        
    finally:
        if os.path.exists(watermarked_path):
            os.unlink(watermarked_path)

def test_verification_profiles(test_video):
    """Test different verification profiles"""
    set_deterministic_seed(42)
    secret = np.random.randint(0, 2, size=32).astype(np.float32)
    
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
        watermarked_path = f.name
    
    try:
        embed_video(test_video, watermarked_path, secret, crf=23)
        
        profiles = ["strict", "balanced", "lenient"]
        results = {}
        
        for profile in profiles:
            result = verify_video(
                watermarked_path, 
                secret, 
                n_frames=5,
                profile=profile
            )
            results[profile] = result
            
            # BER should be the same, but validity may differ
            assert "valid" in result
            assert "ber" in result
        
        # All profiles should have same BER
        bers = [results[p]["ber"] for p in profiles]
        assert all(abs(ber - bers[0]) < 1e-6 for ber in bers), "BER should be same across profiles"
        
        # But validity might differ based on thresholds
        print(f"Profile results: {[(p, results[p]['valid']) for p in profiles]}")
        
    finally:
        if os.path.exists(watermarked_path):
            os.unlink(watermarked_path)

def test_artifact_export(test_video):
    """Test BER artifact export functionality"""
    set_deterministic_seed(42)
    secret = np.random.randint(0, 2, size=32).astype(np.float32)
    
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
        watermarked_path = f.name
    
    with tempfile.TemporaryDirectory() as out_dir:
        try:
            embed_video(test_video, watermarked_path, secret, crf=23)
            
            result = verify_video(
                watermarked_path, 
                secret, 
                n_frames=8,
                export_artifacts=True,
                out_dir=out_dir
            )
            
            # Check that artifacts were created
            csv_path = os.path.join(out_dir, "ber.csv")
            png_path = os.path.join(out_dir, "ber.png")
            
            assert os.path.exists(csv_path), "BER CSV not created"
            assert os.path.exists(png_path), "BER plot not created"
            
            # Check CSV content
            with open(csv_path, 'r') as f:
                lines = f.readlines()
                assert len(lines) > 1, "CSV should have header + data"
                assert lines[0].strip() == "frame,ber", "CSV header incorrect"
            
            # Check that per-frame BER data is reasonable
            assert len(result["per_frame_ber"]) == 8, "Should have 8 frame BER values"
            assert all(0 <= ber <= 1 for ber in result["per_frame_ber"]), "BER values should be in [0,1]"
            
        finally:
            if os.path.exists(watermarked_path):
                os.unlink(watermarked_path)

def test_error_handling_invalid_video():
    """Test error handling with invalid video file"""
    secret = np.random.randint(0, 2, size=32).astype(np.float32)
    
    # Test with non-existent file
    with pytest.raises(ValueError, match="Invalid video file"):
        verify_video("nonexistent.mp4", secret)
    
    # Test with non-video file
    with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
        f.write(b"not a video")
        txt_path = f.name
    
    try:
        with pytest.raises(ValueError, match="Invalid video file"):
            verify_video(txt_path, secret)
    finally:
        os.unlink(txt_path)

def test_reproducibility_with_seed():
    """Test that results are reproducible with same seed"""
    # This test uses a minimal fake video since we're testing reproducibility
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
        # Create minimal MP4 file
        f.write(b'\x00\x00\x00\x20ftypmp4\x00\x00\x00\x00mp41isom')
        f.write(b'\x00' * 1000)
        fake_video = f.name
    
    try:
        # Generate secrets with same seed
        set_deterministic_seed(42)
        secret1 = np.random.randint(0, 2, size=32).astype(np.float32)
        
        set_deterministic_seed(42)
        secret2 = np.random.randint(0, 2, size=32).astype(np.float32)
        
        # Secrets should be identical
        np.testing.assert_array_equal(secret1, secret2)
        
    finally:
        os.unlink(fake_video)

@pytest.mark.slow
def test_performance_benchmark(test_video):
    """Benchmark embed and verify performance"""
    import time
    
    set_deterministic_seed(42)
    secret = np.random.randint(0, 2, size=32).astype(np.float32)
    
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
        watermarked_path = f.name
    
    try:
        # Benchmark embedding
        start_time = time.time()
        embed_video(test_video, watermarked_path, secret, crf=23)
        embed_time = time.time() - start_time
        
        # Benchmark verification
        start_time = time.time()
        result = verify_video(watermarked_path, secret, n_frames=10)
        verify_time = time.time() - start_time
        
        print(f"Embedding time: {embed_time:.2f}s")
        print(f"Verification time: {verify_time:.2f}s")
        
        # Reasonable performance expectations (adjust based on hardware)
        assert embed_time < 30, f"Embedding too slow: {embed_time}s"
        assert verify_time < 10, f"Verification too slow: {verify_time}s"
        
        # Verification should be faster than embedding
        assert verify_time < embed_time, "Verification should be faster than embedding"
        
    finally:
        if os.path.exists(watermarked_path):
            os.unlink(watermarked_path)