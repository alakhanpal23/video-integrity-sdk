#!/usr/bin/env python3
"""
Example: Embed and verify watermark in video

This example demonstrates the complete workflow:
1. Create a test video
2. Embed a watermark
3. Verify the watermark
4. Test robustness against compression
"""

import os
import sys
import tempfile
import numpy as np
import cv2

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from encoder import embed_video
from cli import verify_video
from utils import set_deterministic_seed, save_ber_artifacts

def create_test_video(path: str, duration_sec: int = 3, fps: int = 10, size: int = 256):
    """Create a simple test video with color bars"""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(path, fourcc, fps, (size, size))
    
    total_frames = duration_sec * fps
    
    for i in range(total_frames):
        # Create color bar pattern
        frame = np.zeros((size, size, 3), dtype=np.uint8)
        
        # Red, Green, Blue bars
        frame[:, :size//3] = (255, 0, 0)
        frame[:, size//3:2*size//3] = (0, 255, 0)
        frame[:, 2*size//3:] = (0, 0, 255)
        
        # Add some motion
        frame = np.roll(frame, i * 3, axis=1)
        
        out.write(frame)
    
    out.release()
    print(f"✅ Created test video: {path} ({total_frames} frames)")

def main():
    # Set seed for reproducible results
    set_deterministic_seed(42)
    
    # Generate a test secret (32 bits)
    secret = np.random.randint(0, 2, size=32).astype(np.float32)
    secret_hex = ''.join([str(int(b)) for b in secret])
    print(f"🔐 Generated secret: {secret_hex[:16]}... ({len(secret)} bits)")
    
    # Create temporary files
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
        original_path = f.name
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
        watermarked_path = f.name
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
        compressed_path = f.name
    
    try:
        # Step 1: Create test video
        print("\n📹 Step 1: Creating test video...")
        create_test_video(original_path, duration_sec=2, fps=15)
        
        # Step 2: Embed watermark
        print("\n🔐 Step 2: Embedding watermark...")
        embed_video(original_path, watermarked_path, secret, crf=18)
        print(f"✅ Watermarked video saved: {watermarked_path}")
        
        # Step 3: Verify watermark (original quality)
        print("\n🔍 Step 3: Verifying watermark (original quality)...")
        result_original = verify_video(
            watermarked_path, 
            secret, 
            n_frames=8,
            profile="balanced",
            export_artifacts=True,
            out_dir="out"
        )
        
        print(f"   Valid: {result_original['valid']}")
        print(f"   BER: {result_original['ber']:.4f}")
        print(f"   Summary: {result_original['summary']}")
        
        # Step 4: Test compression robustness
        print("\n📼 Step 4: Testing compression robustness...")
        import subprocess
        
        # Compress with higher CRF (lower quality)
        subprocess.run([
            'ffmpeg', '-i', watermarked_path, '-c:v', 'libx264', 
            '-crf', '28', compressed_path, '-y', '-hide_banner', '-loglevel', 'error'
        ], check=True)
        
        print(f"✅ Compressed video saved: {compressed_path}")
        
        # Verify compressed version
        print("\n🔍 Step 5: Verifying compressed video...")
        result_compressed = verify_video(
            compressed_path, 
            secret, 
            n_frames=8,
            profile="balanced"
        )
        
        print(f"   Valid: {result_compressed['valid']}")
        print(f"   BER: {result_compressed['ber']:.4f}")
        print(f"   Summary: {result_compressed['summary']}")
        
        # Step 6: Compare results
        print("\n📊 Step 6: Results comparison...")
        print(f"   Original BER:   {result_original['ber']:.4f}")
        print(f"   Compressed BER: {result_compressed['ber']:.4f}")
        print(f"   BER increase:   {result_compressed['ber'] - result_original['ber']:.4f}")
        
        if result_compressed['ber'] < 0.2:
            print("✅ Watermark survived compression!")
        else:
            print("⚠️  Watermark degraded significantly")
        
        # Step 7: Test with wrong secret
        print("\n🔍 Step 7: Testing with wrong secret...")
        wrong_secret = np.random.randint(0, 2, size=32).astype(np.float32)
        result_wrong = verify_video(watermarked_path, wrong_secret, n_frames=8)
        
        print(f"   Wrong secret BER: {result_wrong['ber']:.4f}")
        print(f"   Valid: {result_wrong['valid']}")
        
        if result_wrong['ber'] > result_original['ber']:
            print("✅ Wrong secret correctly rejected!")
        
        print(f"\n🎉 Demo completed! Check out/ber.png for BER timeline visualization")
        
        # Try to open the plot
        if os.path.exists("out/ber.png"):
            try:
                import subprocess
                subprocess.run(["open", "out/ber.png"], check=False)  # macOS
            except:
                try:
                    subprocess.run(["xdg-open", "out/ber.png"], check=False)  # Linux
                except:
                    pass  # Windows or other
        
    finally:
        # Cleanup temporary files
        for path in [original_path, watermarked_path, compressed_path]:
            if os.path.exists(path):
                os.unlink(path)

if __name__ == "__main__":
    main()