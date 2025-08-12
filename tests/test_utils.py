# tests/test_utils.py

import pytest
import tempfile
import os
import numpy as np
import torch
from src.utils import (
    save_ber_artifacts, format_summary, validate_video_file, 
    set_deterministic_seed, Timer
)

def test_save_ber_artifacts():
    """Test BER artifact generation"""
    ber_data = [0.05, 0.1, 0.25, 0.15, 0.08, 0.3, 0.12]
    
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path, png_path = save_ber_artifacts(ber_data, tmpdir)
        
        # Check files exist
        assert os.path.exists(csv_path)
        assert os.path.exists(png_path)
        
        # Check CSV content
        with open(csv_path, 'r') as f:
            lines = f.readlines()
            assert lines[0].strip() == "frame,ber"
            assert len(lines) == len(ber_data) + 1  # Header + data
            assert lines[1].strip() == "0,0.05"
            assert lines[-1].strip() == f"{len(ber_data)-1},{ber_data[-1]}"

def test_format_summary():
    """Test summary formatting"""
    ber_data = [0.05, 0.1, 0.25, 0.15, 0.08, 0.3, 0.12]
    thresholds = {"pass": 0.1, "fail": 0.2}
    
    # Valid case
    summary = format_summary(True, 0.08, ber_data, thresholds)
    assert "Valid: Yes" in summary
    assert "avg BER 0.080" in summary
    # Check that high BER frames are detected (exact format may vary)
    assert "frames 2" in summary  # High BER at frame 2 (0.25)
    assert "frames 5" in summary  # High BER at frame 5 (0.3)
    
    # Invalid case
    summary = format_summary(False, 0.25, ber_data, thresholds)
    assert "Invalid: No" in summary
    assert "avg BER 0.250" in summary

def test_validate_video_file():
    """Test video file validation"""
    # Test non-existent file
    assert not validate_video_file("nonexistent.mp4")
    
    # Test with temporary files
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
        f.write(b"fake video content")
        temp_path = f.name
    
    try:
        # Valid extension, small size
        assert validate_video_file(temp_path, max_size_mb=1)
        
        # File too large
        assert not validate_video_file(temp_path, max_size_mb=0)
    finally:
        os.unlink(temp_path)
    
    # Test invalid extension
    with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
        f.write(b"not a video")
        temp_path = f.name
    
    try:
        assert not validate_video_file(temp_path)
    finally:
        os.unlink(temp_path)

def test_deterministic_seed():
    """Test deterministic seed setting"""
    set_deterministic_seed(42)
    
    # Test numpy
    rand1 = np.random.random(5)
    set_deterministic_seed(42)
    rand2 = np.random.random(5)
    np.testing.assert_array_equal(rand1, rand2)
    
    # Test torch
    set_deterministic_seed(42)
    tensor1 = torch.randn(3, 3)
    set_deterministic_seed(42)
    tensor2 = torch.randn(3, 3)
    torch.testing.assert_close(tensor1, tensor2)

def test_timer():
    """Test Timer context manager"""
    import time
    
    with Timer("Test operation") as timer:
        time.sleep(0.1)
    
    # Timer should have recorded the operation
    assert timer.start_time is not None

def test_format_summary_edge_cases():
    """Test summary formatting edge cases"""
    thresholds = {"pass": 0.1, "fail": 0.2}
    
    # No high BER runs
    ber_data = [0.05, 0.08, 0.06, 0.09]
    summary = format_summary(True, 0.07, ber_data, thresholds)
    assert "Potential edits" not in summary
    
    # Continuous high BER run
    ber_data = [0.05, 0.25, 0.3, 0.28, 0.08]
    summary = format_summary(False, 0.2, ber_data, thresholds)
    assert "frames 1-3" in summary
    
    # Run at the end
    ber_data = [0.05, 0.08, 0.25, 0.3]
    summary = format_summary(False, 0.17, ber_data, thresholds)
    assert "frames 2-3" in summary