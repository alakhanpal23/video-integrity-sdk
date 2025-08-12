# tests/test_video_analysis.py

import pytest
import torch
import numpy as np
from src.video_analysis import analyze_frames

def test_analyze_frames_basic():
    """Test basic frame analysis"""
    # Create test frames [N, 3, H, W]
    frames = torch.randn(5, 3, 64, 64)
    
    result = analyze_frames(frames)
    
    assert isinstance(result, dict)
    assert result['num_frames'] == 5
    assert 'warnings' in result
    assert 'mean_pixel' in result
    assert len(result['mean_pixel']) == 5

def test_analyze_frames_with_ber():
    """Test frame analysis with BER data"""
    frames = torch.randn(3, 3, 64, 64)
    per_frame_ber = np.array([0.1, 0.6, 0.2])  # High BER in frame 1
    
    result = analyze_frames(frames, per_frame_ber=per_frame_ber)
    
    assert result['num_frames'] == 3
    assert len(result['tampered_frames']) == 1
    assert 1 in result['tampered_frames']

def test_analyze_frames_all_black():
    """Test detection of all-black frames"""
    frames = torch.zeros(2, 3, 64, 64)  # All black
    
    result = analyze_frames(frames)
    
    assert result['all_black'] == 2
    assert len([w for w in result['warnings'] if 'all black' in w]) == 2

def test_analyze_frames_all_white():
    """Test detection of all-white frames"""
    frames = torch.ones(2, 3, 64, 64)  # All white
    
    result = analyze_frames(frames)
    
    assert result['all_white'] == 2
    assert len([w for w in result['warnings'] if 'all white' in w]) == 2

def test_analyze_frames_mixed():
    """Test analysis with mixed frame types"""
    frames = torch.stack([
        torch.zeros(3, 64, 64),      # All black
        torch.ones(3, 64, 64),       # All white  
        torch.randn(3, 64, 64) * 0.1 + 0.5,  # Normal
    ])
    
    result = analyze_frames(frames)
    
    assert result['num_frames'] == 3
    assert result['all_black'] == 1
    assert result['all_white'] == 1
    assert len(result['warnings']) >= 2

def test_analyze_frames_empty():
    """Test analysis with no frames"""
    frames = torch.empty(0, 3, 64, 64)
    
    result = analyze_frames(frames)
    
    assert result['num_frames'] == 0
    assert len(result['mean_pixel']) == 0

def test_analyze_frames_statistics():
    """Test that statistics are computed correctly"""
    # Create frames with known properties
    frames = torch.stack([
        torch.full((3, 64, 64), 0.2),  # Low brightness
        torch.full((3, 64, 64), 0.8),  # High brightness
        torch.full((3, 64, 64), 0.5),  # Medium brightness
    ])
    
    result = analyze_frames(frames)
    
    # Check mean pixel values
    assert abs(result['mean_pixel'][0] - 0.2) < 0.01
    assert abs(result['mean_pixel'][1] - 0.8) < 0.01
    assert abs(result['mean_pixel'][2] - 0.5) < 0.01
    
    # Check blur scores exist
    assert len(result['blur_score']) == 3
    assert all(isinstance(score, float) for score in result['blur_score'])
    
    # Check noise std exists
    assert len(result['noise_std']) == 3
    assert all(isinstance(std, float) for std in result['noise_std'])