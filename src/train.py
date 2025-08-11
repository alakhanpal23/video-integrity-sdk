import os
import glob
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import ffmpeg
from tqdm import tqdm
from encoder import WatermarkEncoder
from decoder import WatermarkDecoder

# --- Helper functions ---
def extract_frames(video_path, num_frames=16, size=256):
    """Extracts num_frames uniformly from video as [num_frames, 3, size, size] np.float32 in [0,1]."""
    # Probe for total frames
    probe = ffmpeg.probe(video_path)
    video_stream = next(s for s in probe['streams'] if s['codec_type'] == 'video')
    total_frames = int(video_stream.get('nb_frames', 0))
    if total_frames == 0:
        # fallback: estimate from duration and r_frame_rate
        duration = float(video_stream['duration'])
        fps = eval(video_stream['r_frame_rate'])
        total_frames = int(duration * fps)
    steps = np.linspace(0, max(0, total_frames-1), num_frames, dtype=int)
    frames = []
    for idx in steps:
        out, _ = (
            ffmpeg.input(video_path)
            .filter('select', f'eq(n\,{idx}')
            .output('pipe:', vframes=1, format='rawvideo', pix_fmt='rgb24', s=f'{size}x{size}')
            .run(capture_stdout=True, quiet=True)
        )
        frame = np.frombuffer(out, np.uint8).reshape([size, size, 3])
        frame = frame.astype(np.float32) / 255.0
        frame = np.transpose(frame, (2,0,1))  # [3,H,W]
        frames.append(frame)
    return np.stack(frames)  # [N,3,H,W]

def save_checkpoint(encoder, decoder, optimizer, epoch, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    torch.save({
        'encoder': encoder.state_dict(),
        'decoder': decoder.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch
    }, os.path.join(out_dir, f'checkpoint_{epoch}.pt'))

# --- Training script ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='data', help='Directory with training videos')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--alpha', type=float, default=0.1, help='Weight for MSE loss')
    parser.add_argument('--num-frames', type=int, default=8, help='Frames per video per batch')
    parser.add_argument('--secret-len', type=int, default=32)
    parser.add_argument('--checkpoint-dir', type=str, default='models')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder = WatermarkEncoder(secret_len=args.secret_len).to(device)
    decoder = WatermarkDecoder(secret_len=args.secret_len).to(device)
    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=args.lr)
    bce = nn.BCEWithLogitsLoss()
    mse = nn.MSELoss()

    # Gather all video files
    video_files = glob.glob(os.path.join(args.data_dir, '*.mp4'))
    if not video_files:
        raise RuntimeError(f'No videos found in {args.data_dir}')

    for epoch in range(1, args.epochs+1):
        np.random.shuffle(video_files)
        pbar = tqdm(video_files, desc=f'Epoch {epoch}')
        for vid in pbar:
            # Extract frames
            frames = extract_frames(vid, num_frames=args.num_frames, size=256)  # [N,3,256,256]
            frames = torch.from_numpy(frames).float().to(device)
            for i in range(0, len(frames), args.batch_size):
                batch = frames[i:i+args.batch_size]
                B = batch.shape[0]
                # Generate random secrets
                secrets = torch.randint(0, 2, (B, args.secret_len), dtype=torch.float32, device=device)
                # Encode
                encoded = encoder(batch, secrets)
                # Add small Gaussian noise to simulate codec artifacts
                noisy = encoded + 0.01 * torch.randn_like(encoded)
                # Decode
                logits = decoder(noisy)
                # Loss: bit recovery + perceptual
                loss_bce = bce(logits, secrets)
                loss_mse = mse(encoded, batch)
                loss = loss_bce + args.alpha * loss_mse
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                pbar.set_postfix({'loss': loss.item(), 'bce': loss_bce.item(), 'mse': loss_mse.item()})
        save_checkpoint(encoder, decoder, optimizer, epoch, args.checkpoint_dir)
        print(f'Checkpoint saved for epoch {epoch}')

if __name__ == '__main__':
    main() 