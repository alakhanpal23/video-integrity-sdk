# utils.py: Utility functions for BER analysis, progress tracking, and artifacts

import csv
import os
import time
from typing import List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch

def save_ber_artifacts(ber_per_frame: List[float], out_dir: str = "out") -> Tuple[str, str]:
    """Save BER timeline as CSV and PNG plot"""
    os.makedirs(out_dir, exist_ok=True)
    
    csv_path = os.path.join(out_dir, "ber.csv")
    png_path = os.path.join(out_dir, "ber.png")
    
    # Save CSV
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "ber"])
        for i, ber in enumerate(ber_per_frame):
            writer.writerow([i, ber])
    
    # Save plot
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(ber_per_frame)), ber_per_frame, 'b-', linewidth=2)
    plt.axhline(y=0.1, color='g', linestyle='--', alpha=0.7, label='Pass threshold')
    plt.axhline(y=0.2, color='r', linestyle='--', alpha=0.7, label='Fail threshold')
    plt.xlabel("Frame Index")
    plt.ylabel("Bit Error Rate (BER)")
    plt.title("BER Timeline - Video Integrity Analysis")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(png_path, dpi=160, bbox_inches='tight')
    plt.close()
    
    return csv_path, png_path

def format_summary(valid: bool, avg_ber: float, ber_per_frame: List[float], 
                  thresholds: dict) -> str:
    """Generate human-readable verification summary"""
    status = "Valid" if valid else "Invalid"
    
    # Find high BER runs (potential tampering)
    high_ber_runs = []
    in_run = False
    run_start = 0
    
    for i, ber in enumerate(ber_per_frame):
        if ber > thresholds["fail"] and not in_run:
            in_run = True
            run_start = i
        elif ber <= thresholds["pass"] and in_run:
            in_run = False
            high_ber_runs.append((run_start, i-1))
    
    if in_run:  # Run continues to end
        high_ber_runs.append((run_start, len(ber_per_frame)-1))
    
    summary = f"{status}: {'Yes' if valid else 'No'} (avg BER {avg_ber:.3f})"
    
    if high_ber_runs:
        runs_str = ", ".join([f"frames {start}-{end}" for start, end in high_ber_runs])
        summary += f". Potential edits at {runs_str} (high BER runs)"
    
    return summary

def progress_wrapper(iterable, desc: str = "Processing", disable: bool = False):
    """Wrapper for tqdm progress bars"""
    return tqdm(iterable, desc=desc, disable=disable, 
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')

class Timer:
    """Context manager for timing operations"""
    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time = None
        
    def __enter__(self):
        self.start_time = time.time()
        return self
        
    def __exit__(self, *args):
        duration = time.time() - self.start_time
        print(f"{self.name} completed in {duration:.2f}s")

def validate_video_file(file_path: str, max_size_mb: int = 100) -> bool:
    """Validate video file size and basic properties"""
    if not os.path.exists(file_path):
        return False
        
    # Check file size
    size_mb = os.path.getsize(file_path) / (1024 * 1024)
    if size_mb > max_size_mb:
        return False
        
    # Basic extension check
    valid_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
    _, ext = os.path.splitext(file_path.lower())
    return ext in valid_extensions

def set_deterministic_seed(seed: int):
    """Set seeds for reproducible results"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)