#!/usr/bin/env python3
"""
Example: Verify video via API

This example demonstrates how to use the REST API for verification.
"""

import requests
import tempfile
import os
import cv2
import numpy as np

def create_test_video(path: str):
    """Create a simple test video"""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(path, fourcc, 10.0, (256, 256))
    
    for i in range(30):
        frame = np.zeros((256, 256, 3), dtype=np.uint8)
        frame[:, :85] = (255, 0, 0)
        frame[:, 85:170] = (0, 255, 0)
        frame[:, 170:] = (0, 0, 255)
        frame = np.roll(frame, i * 5, axis=1)
        out.write(frame)
    
    out.release()

def main():
    api_url = "http://127.0.0.1:8000"
    
    # Check if API is running
    try:
        health_response = requests.get(f"{api_url}/healthz", timeout=5)
        if health_response.status_code != 200:
            print("❌ API health check failed")
            return
        print("✅ API is healthy")
    except requests.exceptions.RequestException:
        print("❌ Cannot connect to API. Make sure the server is running:")
        print("   uvicorn src.api:app --host 0.0.0.0 --port 8000")
        return
    
    # Create test video
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
        video_path = f.name
    
    try:
        print("📹 Creating test video...")
        create_test_video(video_path)
        
        # Test secret
        secret_hex = "ffeeddccbbaa99887766554433221100"
        
        # Verify via API
        print("🔍 Verifying via API...")
        
        with open(video_path, 'rb') as f:
            files = {"file": ("test.mp4", f, "video/mp4")}
            data = {
                "secret_hex": secret_hex,
                "n_frames": 8,
                "profile": "balanced",
                "export_artifacts": "true"
            }
            
            # Add API key if set
            headers = {}
            api_key = os.getenv("API_KEY")
            if api_key:
                headers["x-api-key"] = api_key
                print(f"🔐 Using API key: {api_key[:8]}...")
            
            response = requests.post(
                f"{api_url}/verify", 
                files=files, 
                data=data,
                headers=headers,
                timeout=30
            )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Verification successful!")
            print(f"   Valid: {result['valid']}")
            print(f"   BER: {result['ber']:.4f}")
            print(f"   Summary: {result['summary']}")
            print(f"   Processing time: {result['processing_time']:.2f}s")
            
            if result.get('artifacts'):
                print("📊 Artifacts exported (base64 encoded)")
                
        elif response.status_code == 401:
            print("❌ Authentication failed. Set API_KEY environment variable:")
            print("   export API_KEY=your-api-key")
            
        else:
            print(f"❌ Verification failed: {response.status_code}")
            print(f"   Error: {response.json().get('detail', 'Unknown error')}")
    
    finally:
        if os.path.exists(video_path):
            os.unlink(video_path)

if __name__ == "__main__":
    main()