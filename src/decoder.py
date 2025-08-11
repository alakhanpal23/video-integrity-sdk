# src/decoder.py

import torch
import torch.nn as nn
import numpy as np

"""
WatermarkDecoder module.

Input: frame [B, 3, H, W] (float32, range [0,1])
Output: logits [B, secret_len] (float32, unbounded)
"""
class WatermarkDecoder(nn.Module):
    def __init__(self, secret_len: int = 32):
        super().__init__()
        # simple two‐layer CNN to extract residual
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 3, kernel_size=3, padding=1)
        # collapse spatial features back into secret bits
        self.fc = nn.Linear(3 * 64 * 64, secret_len)

    def forward(self, frame: torch.Tensor) -> torch.Tensor:
        """
        frame: [B,3,H,W] (float32, range [0,1])
        returns: logits for the secret bits [B, secret_len]
        """
        x = torch.relu(self.conv1(frame))
        x = torch.relu(self.conv2(x))
        # Pool to 64x64
        x = torch.nn.functional.adaptive_avg_pool2d(x, (64, 64))
        B, C, H, W = x.shape
        x = x.view(B, C * 64 * 64)
        return self.fc(x)

    @staticmethod
    def decode_bits(logits: torch.Tensor, threshold: float = 0.5) -> np.ndarray:
        """
        logits: [B, secret_len] (float32, unbounded)
        threshold: float, threshold for bit decision after sigmoid
        returns: binary array [B, secret_len] (np.uint8)
        """
        probs = torch.sigmoid(logits)
        bits = (probs > threshold).cpu().numpy().astype(np.uint8)
        return bits
