import torch
import numpy as np
from video_analysis import analyze_frames

def test_all_black_white():
    frames = torch.zeros((2, 3, 256, 256), dtype=torch.float32)
    frames[1] = 1.0
    report = analyze_frames(frames)
    assert report['all_black'] == 1
    assert report['all_white'] == 1
    assert any('all black' in w for w in report['warnings'])
    assert any('all white' in w for w in report['warnings'])

def test_blur_noise_hist():
    # Blurry frame: flat gray
    blurry = torch.full((1, 3, 256, 256), 0.5)
    # Noisy frame
    noisy = torch.clamp(torch.randn((1, 3, 256, 256)) * 0.3 + 0.5, 0, 1)
    frames = torch.cat([blurry, noisy], dim=0)
    report = analyze_frames(frames)
    assert report['blurred'] >= 1
    assert report['noisy'] >= 1
    assert report['hist_outlier'] >= 1

def test_tamper_localization():
    frames = torch.rand((3, 3, 256, 256))
    per_frame_ber = np.array([0.1, 0.6, 0.8])
    report = analyze_frames(frames, per_frame_ber=per_frame_ber)
    assert 1 in report['tampered_frames']
    assert 2 in report['tampered_frames']
    assert any('likely tampered' in w for w in report['warnings']) 