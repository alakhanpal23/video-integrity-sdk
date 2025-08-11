# video_analysis.py: Video frame analysis utilities
#
# - analyze_frames(frames, per_frame_ber=None): Checks for corrupted, all-black, all-white, blur, noise, histogram outliers.
#   Optionally flags frames with high per-frame BER as likely tampered.
#   Returns a summary report (dict).
#
# To extend, add more checks (e.g., blockiness, color shifts, region-based tamper localization).

import torch
import numpy as np
import cv2

def analyze_frames(frames: torch.Tensor, per_frame_ber=None):
    """
    frames: [N, 3, H, W] float32 in [0,1]
    per_frame_ber: Optional [N] array of BERs for tamper localization
    Returns: dict with basic stats and warnings.
    """
    N = frames.shape[0]
    stats = {
        'num_frames': N,
        'all_black': 0,
        'all_white': 0,
        'blurred': 0,
        'noisy': 0,
        'hist_outlier': 0,
        'mean_pixel': [],
        'blur_score': [],
        'noise_std': [],
        'warnings': [],
        'tampered_frames': []
    }
    for i in range(N):
        f = frames[i].cpu().numpy()
        mean = float(np.mean(f))
        stats['mean_pixel'].append(mean)
        # All black/white
        if np.allclose(f, 0, atol=1e-3):
            stats['all_black'] += 1
            stats['warnings'].append(f'Frame {i} is all black')
        if np.allclose(f, 1, atol=1e-3):
            stats['all_white'] += 1
            stats['warnings'].append(f'Frame {i} is all white')
        # Blur (variance of Laplacian)
        img = (np.transpose(f, (1,2,0)) * 255).astype(np.uint8)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        stats['blur_score'].append(blur_score)
        if blur_score < 20:  # Threshold for blur (tune as needed)
            stats['blurred'] += 1
            stats['warnings'].append(f'Frame {i} is blurry (score={blur_score:.1f})')
        # Noise (stddev)
        noise_std = float(np.std(f))
        stats['noise_std'].append(noise_std)
        if noise_std > 0.25:  # Threshold for noise (tune as needed)
            stats['noisy'] += 1
            stats['warnings'].append(f'Frame {i} is noisy (std={noise_std:.2f})')
        # Histogram outlier (flat or spiky)
        hist = cv2.calcHist([gray], [0], None, [256], [0,256])
        hist = hist.flatten() / hist.sum()
        if np.max(hist) > 0.25 or np.min(hist) < 1e-4:
            stats['hist_outlier'] += 1
            stats['warnings'].append(f'Frame {i} has histogram outlier')
        # Tamper localization
        if per_frame_ber is not None and per_frame_ber[i] > 0.5:
            stats['tampered_frames'].append(i)
            stats['warnings'].append(f'Frame {i} likely tampered (BER={per_frame_ber[i]:.2f})')
    return stats 