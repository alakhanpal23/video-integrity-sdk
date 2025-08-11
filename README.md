# Video Integrity SDK

A Python/FFmpeg-based SDK to embed and verify imperceptible watermarks in video streams.

## Quickstart

```bash
git clone <repo-url>
cd video-integrity-sdk
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Embed a Watermark

```bash
# Generate a test video (color bars)
ffmpeg -f lavfi -i testsrc=duration=2:size=256x256:rate=10 in.mp4 -y

# Embed a deterministic watermark (hex secret)
python src/encoder.py --embed in.mp4 out.mp4 --secret ffeeddccbbaa99887766554433221100 --crf 18
```

### Run the Verification Server

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

### Verify a Watermarked Video

```bash
curl -F "file=@out.mp4" -F "secret_hex=ffeeddccbbaa99887766554433221100" -F "n_frames=8" http://127.0.0.1:8000/verify
```

Or in Python:

```python
import requests
with open("out.mp4", "rb") as f:
    files = {"file": ("video.mp4", f, "video/mp4")}
    data = {"secret_hex": "ffeeddccbbaa99887766554433221100", "n_frames": 8}
    resp = requests.post("http://127.0.0.1:8000/verify", files=files, data=data)
    print(resp.json())
```

## CLI Usage

```bash
python src/encoder.py --embed <in.mp4> <out.mp4> [--secret <hex>] [--crf <int>]

# Examples:
python src/encoder.py --embed in.mp4 out.mp4
python src/encoder.py --embed in.mp4 out.mp4 --secret ffeeddccbbaa99887766554433221100
python src/encoder.py --embed in.mp4 out.mp4 --crf 20
```

- `--embed <in> <out>`: Input and output video files
- `--secret <hex>`: 32-bit secret as hex string (deterministic watermark)
- `--crf <int>`: Output quality (lower is better, default 23)

## Watermark Robustness & BER

- The watermark is designed to survive typical video compression (e.g., H.264, CRF 18-28).
- **Bit Error Rate (BER)** is the fraction of secret bits incorrectly recovered. Lower is better:
  - `ber < 0.1`: Watermark is considered valid (robust)
  - `ber > 0.2`: Watermark is likely lost or tampered
- Corrupting or blacking out frames increases BER, but the watermark is robust to moderate compression and re-encoding.

## Further Documentation

- [Design Rationale](docs/design.md) (coming soon)
- [API Reference](src/api.py)
- [Encoder/Decoder Details](src/encoder.py, src/decoder.py)

For questions or contributions, see [CONTRIBUTING.md](CONTRIBUTING.md) or open an issue.
