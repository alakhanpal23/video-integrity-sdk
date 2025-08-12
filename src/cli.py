# cli.py: Enhanced CLI with verification support

import argparse
import sys
import os
import numpy as np
import torch
import cv2
import subprocess
import tempfile

try:
    from .decoder import WatermarkDecoder
    from .config import load_config, get_profile_thresholds
    from .utils import save_ber_artifacts, format_summary, progress_wrapper, Timer, set_deterministic_seed, validate_video_file
    from .video_analysis import analyze_frames
except ImportError:
    from decoder import WatermarkDecoder
    from config import load_config, get_profile_thresholds
    from utils import save_ber_artifacts, format_summary, progress_wrapper, Timer, set_deterministic_seed, validate_video_file
    from video_analysis import analyze_frames

def verify_video(input_path: str, secret: np.ndarray, n_frames: int = 10, 
                profile: str = "balanced", sample_strategy: str = "uniform",
                export_artifacts: bool = False, out_dir: str = "out"):
    """CLI video verification with BER analysis"""
    
    if not validate_video_file(input_path):
        raise ValueError(f"Invalid video file: {input_path}")
    
    thresholds = get_profile_thresholds(profile)
    
    with Timer("Video verification"):
        # Get total frame count
        probe = subprocess.run([
            'ffprobe', '-v', 'error', '-select_streams', 'v:0',
            '-count_frames', '-show_entries', 'stream=nb_read_frames',
            '-of', 'default=nokey=1:noprint_wrappers=1', input_path
        ], capture_output=True, text=True)
        total_frames = int(probe.stdout.strip() or 0)
        
        if total_frames == 0:
            # fallback
            probe2 = subprocess.run([
                'ffprobe', '-v', 'error', '-select_streams', 'v:0',
                '-show_entries', 'stream=nb_frames',
                '-of', 'default=nokey=1:noprint_wrappers=1', input_path
            ], capture_output=True, text=True)
            total_frames = int(probe2.stdout.strip() or 0)
            
        if total_frames == 0:
            raise ValueError("Could not determine frame count")
            
        # Sample frames based on strategy
        if sample_strategy == "uniform":
            step = max(1, total_frames // n_frames)
        elif sample_strategy == "keyframes":
            step = max(1, total_frames // (n_frames * 2))  # More frequent sampling
        else:  # stride
            step = int(sample_strategy.split('=')[1]) if '=' in sample_strategy else 5
            
        # Extract frames
        frames_dir = tempfile.mkdtemp()
        out_pattern = os.path.join(frames_dir, 'frame_%04d.png')
        ffmpeg_cmd = [
            'ffmpeg', '-i', input_path,
            '-vf', f'select=not(mod(n\\,{step}))',
            '-vsync', 'vfr', '-frames:v', str(n_frames), out_pattern,
            '-hide_banner', '-loglevel', 'error'
        ]
        subprocess.run(ffmpeg_cmd, check=True)
        
        # Load frames
        frame_files = sorted([os.path.join(frames_dir, f) for f in os.listdir(frames_dir) if f.endswith('.png')])
        if not frame_files:
            raise ValueError("No frames extracted")
            
        batch = []
        for fpath in progress_wrapper(frame_files, desc="Loading frames"):
            img = cv2.imread(fpath, cv2.IMREAD_COLOR)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (256, 256))
            img = img.astype(np.float32) / 255.0
            img = np.transpose(img, (2,0,1))
            batch.append(img)
            
        if not batch:
            raise ValueError("No valid frames loaded")
            
        batch_tensor = torch.from_numpy(np.stack(batch)).float()
        
        # Decode watermark
        decoder = WatermarkDecoder(secret_len=secret.shape[0])
        decoder.eval()
        
        with torch.no_grad():
            logits = decoder(batch_tensor)
        bits = WatermarkDecoder.decode_bits(logits)
        
        # Per-frame BER
        per_frame_ber = np.mean(bits != secret, axis=1)
        
        # Majority vote
        rec_bits = (np.sum(bits, axis=0) > (len(bits)//2)).astype(np.uint8)
        ber = float(np.mean(rec_bits != secret))
        valid = ber < thresholds["pass"]
        
        # Enhanced analysis
        analysis_report = analyze_frames(batch_tensor, per_frame_ber=per_frame_ber)
        
        # Generate summary
        summary = format_summary(valid, ber, per_frame_ber.tolist(), thresholds)
        
        # Export artifacts
        if export_artifacts:
            csv_path, png_path = save_ber_artifacts(per_frame_ber.tolist(), out_dir)
            print(f"📊 BER timeline saved: {csv_path}, {png_path}")
        
        # Cleanup
        for f in frame_files:
            os.remove(f)
        os.rmdir(frames_dir)
        
        return {
            "valid": valid,
            "ber": ber,
            "summary": summary,
            "per_frame_ber": per_frame_ber.tolist(),
            "analysis": analysis_report
        }

def main():
    parser = argparse.ArgumentParser(description="Video Integrity SDK CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Verify command
    verify_parser = subparsers.add_parser("verify", help="Verify watermarked video")
    verify_parser.add_argument("input", help="Input video file")
    verify_parser.add_argument("--secret", type=str, required=True, help="Secret as hex string")
    verify_parser.add_argument("--n-frames", type=int, default=10, help="Number of frames to sample")
    verify_parser.add_argument("--profile", choices=["strict", "balanced", "lenient"], 
                              default="balanced", help="Verification profile")
    verify_parser.add_argument("--sample-strategy", choices=["uniform", "keyframes"], 
                              default="uniform", help="Frame sampling strategy")
    verify_parser.add_argument("--export-artifacts", action="store_true", 
                              help="Export BER CSV and plot")
    verify_parser.add_argument("--out-dir", default="out", help="Output directory for artifacts")
    verify_parser.add_argument("--seed", type=int, help="Random seed")
    verify_parser.add_argument("--config", type=str, help="Config file path")
    verify_parser.add_argument("--summary", action="store_true", help="Show human-readable summary")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Load config
    config = load_config(args.config) if hasattr(args, 'config') and args.config else load_config()
    
    # Set seed if provided
    if hasattr(args, 'seed') and args.seed:
        set_deterministic_seed(args.seed)
    
    if args.command == "verify":
        try:
            # Parse secret
            secret_bytes = bytes.fromhex(args.secret)
            secret = np.unpackbits(np.frombuffer(secret_bytes, dtype=np.uint8)).astype(np.float32)
            if secret.shape[0] > 32:
                secret = secret[:32]
            elif secret.shape[0] < 32:
                secret = np.pad(secret, (0, 32 - secret.shape[0]), 'constant')
            
            # Verify video
            result = verify_video(
                args.input, secret, 
                n_frames=args.n_frames,
                profile=args.profile,
                sample_strategy=args.sample_strategy,
                export_artifacts=args.export_artifacts,
                out_dir=args.out_dir
            )
            
            if args.summary:
                print(f"\n{result['summary']}")
                if result['analysis']['warnings']:
                    print("\n⚠️  Warnings:")
                    for warning in result['analysis']['warnings'][:5]:  # Limit output
                        print(f"   {warning}")
            else:
                print(f"Valid: {result['valid']}")
                print(f"BER: {result['ber']:.4f}")
                print(f"Frames analyzed: {len(result['per_frame_ber'])}")
                
        except Exception as e:
            print(f"❌ Verification failed: {e}", file=sys.stderr)
            sys.exit(1)

if __name__ == "__main__":
    main()