# Design Rationale (Stub)

This document will describe the design decisions, architecture, and algorithms used in the Video Integrity SDK.

## Overview
- Embeds imperceptible watermarks in video using Python and FFmpeg.
- Robust to typical video compression and moderate tampering.

## Key Components
- `encoder.py`: Embeds watermark into video frames.
- `decoder.py`: Extracts and verifies watermark.
- `api.py`: FastAPI server for verification API.
- `keys.py`: Key management for watermarking secrets.

## Future Work
- Add more details on watermarking algorithm.
- Document API endpoints and expected inputs/outputs.
- Add diagrams and flowcharts.

_This is a stub. Please expand as the project evolves._
