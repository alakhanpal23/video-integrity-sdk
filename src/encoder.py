# src/encoder.py
#
# - WatermarkEncoder: CNN model to embed a secret bit-vector into video frames.
# - CLI subcommands:
#   * embed: Embed watermark into video (see embed_video)
#   * train: (calls train.py)
#   * derive-key: Derive per-video secret from master key (see keys.py)
#
# To change the watermarking model, edit WatermarkEncoder.
# To add new CLI features, extend the argparse subparsers below.
# To change embedding logic, edit embed_video().
#
# For API/JS integration, see src/api.py and js-sdk/.

import argparse
import numpy as np
import torch
import torch.nn as nn
import keys
import cv2

class WatermarkEncoder(nn.Module):
    def __init__(self, secret_len: int = 32):
        super().__init__()
        # simple two‐layer CNN to produce a tiny residual mask
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 3, kernel_size=3, padding=1)
        # fully‐connected to expand the secret into a feature map
        self.secret_fc = nn.Linear(secret_len, 3 * 64 * 64)

    def forward(self, frame: torch.Tensor, secret: torch.Tensor) -> torch.Tensor:
        """
        frame: [B,3,H,W], secret: [B, secret_len]
        returns: watermarked frame
        """
        # basic feature extraction
        x = torch.relu(self.conv1(frame))
        mask = torch.sigmoid(self.conv2(x))         # residual mask
        B, C, H, W = frame.shape
        # map secret bits into a small spatial map
        secret_map = self.secret_fc(secret)         # [B, 3*64*64]
        secret_map = secret_map.view(B, C, 64, 64)
        # upsample to frame size
        secret_map = torch.nn.functional.interpolate(secret_map, (H, W))
        # add a tiny bit of mask + secret to the frame
        return frame + 0.01 * mask + 0.001 * secret_map

def embed_video(input_path: str, output_path: str, secret: np.ndarray, crf: int = 23):
    """
    Use OpenCV to extract frames, watermark, and write video.
    """
    encoder = WatermarkEncoder(secret_len=secret.shape[0])
    encoder.eval()
    # Open input video
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_tensor = torch.from_numpy(frame_rgb).permute(2,0,1).unsqueeze(0).float() / 255.0
        secret_tensor = torch.from_numpy(secret).unsqueeze(0).float()
        with torch.no_grad():
            watermarked = encoder(frame_tensor, secret_tensor)
        watermarked = (watermarked.squeeze(0).permute(1,2,0).numpy() * 255.0).clip(0,255).astype(np.uint8)
        frames.append(watermarked)
    cap.release()
    # Write output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    for frame in frames:
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    out.release()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")

    embed_parser = subparsers.add_parser("embed", help="Embed watermark")
    embed_parser.add_argument("infile")
    embed_parser.add_argument("outfile")
    embed_parser.add_argument("--secret", type=str, default=None)
    embed_parser.add_argument("--crf", type=int, default=23)

    train_parser = subparsers.add_parser("train", help="Train model")
    train_parser.add_argument("--data-dir", type=str, default="data")
    train_parser.add_argument("--epochs", type=int, default=10)
    train_parser.add_argument("--batch-size", type=int, default=8)
    train_parser.add_argument("--lr", type=float, default=1e-4)

    derive_parser = subparsers.add_parser("derive-key", help="Derive secret for video_id")
    derive_parser.add_argument("video_id", type=str)

    args = parser.parse_args()

    if args.command == "embed":
        in_vid, out_vid = args.infile, args.outfile
        if args.secret:
            secret_bytes = bytes.fromhex(args.secret)
            secret = np.unpackbits(np.frombuffer(secret_bytes, dtype=np.uint8)).astype(np.float32)
            if secret.shape[0] > 32:
                secret = secret[:32]
            elif secret.shape[0] < 32:
                secret = np.pad(secret, (0, 32 - secret.shape[0]), 'constant')
        else:
            secret = np.random.randint(0, 2, size=32).astype(np.float32)
        embed_video(in_vid, out_vid, secret, crf=args.crf)
        print(f"✅ Embedded video written to {out_vid}")
    elif args.command == "train":
        import train
        train.main()
    elif args.command == "derive-key":
        print(keys.derive_secret(args.video_id))
