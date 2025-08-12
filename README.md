# Video Integrity SDK

[![CI](https://github.com/alakhanpal23/video-integrity-sdk/workflows/CI/badge.svg)](https://github.com/alakhanpal23/video-integrity-sdk/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://hub.docker.com)

A production-ready Python/FFmpeg-based SDK to embed and verify imperceptible watermarks in video streams for integrity verification.

## 🚀 Run in 60 Seconds

```bash
# 1. Clone and setup
git clone https://github.com/alakhanpal23/video-integrity-sdk.git
cd video-integrity-sdk
pip install -r requirements.txt

# 2. Quick test (no FFmpeg required)
python test_quick.py

# 3. Full demo (requires FFmpeg)
make demo
```

## ✨ Key Features

- 🔐 **CNN-based watermarking** - Imperceptible, compression-resistant
- 📊 **BER analysis** - Timeline plots and tamper localization  
- 🚀 **Production ready** - Health checks, metrics, monitoring
- 🌐 **Multi-language** - Python API + JavaScript SDK
- ⚡ **Quick presets** - `--preset reencode_light|heavy|social`
- 🎯 **Flexible profiles** - `--profile strict|balanced|lenient`
- 📈 **Observability** - Prometheus metrics, structured logging
- 🐳 **Docker ready** - Complete monitoring stack

### Embed a Watermark

```bash
# With attack simulation preset
python src/encoder.py embed input.mp4 output.mp4 --preset reencode_heavy --seed 42

# Traditional approach
python src/encoder.py embed input.mp4 output.mp4 --secret ffeeddccbbaa99887766554433221100 --crf 18
```

### Run the API Server

```bash
# Development
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload

# Production with authentication
export API_KEY=your-secret-key
uvicorn src.api:app --host 0.0.0.0 --port 8000 --workers 4
```

### Verify a Watermarked Video

```bash
# CLI with human-readable summary
python src/cli.py verify output.mp4 --secret ffeeddccbbaa99887766554433221100 --summary --export-artifacts

# API with authentication
curl -F "file=@output.mp4" -F "secret_hex=ffeeddccbbaa99887766554433221100" \
     -F "profile=balanced" -F "export_artifacts=true" \
     -H "x-api-key: your-api-key" http://127.0.0.1:8000/verify
```

### JavaScript SDK

```javascript
const { verifyVideo } = require('./js-sdk');

// Automatic API key from environment
process.env.VIDEO_SDK_API_KEY = 'your-api-key';

const result = await verifyVideo('video.mp4', 'abc123', 8);
console.log(`Valid: ${result.valid}, BER: ${result.ber.toFixed(4)}`);
console.log(result.summary);
```

## 🛠️ Advanced Usage

### Configuration File

```yaml
# config.yaml
embed:
  crf: 20
  preset: "reencode_light"
verify:
  profile: "strict"
  export_artifacts: true
api:
  cors_origins: ["https://myapp.com"]
  max_file_size_mb: 200
```

```bash
python src/cli.py verify video.mp4 --config config.yaml --summary
```

### Attack Simulation Presets

```bash
# Light compression (CRF 23)
python src/encoder.py embed input.mp4 output.mp4 --preset reencode_light

# Heavy compression (CRF 35) 
python src/encoder.py embed input.mp4 output.mp4 --preset reencode_heavy

# Social media optimization (CRF 28, scaled)
python src/encoder.py embed input.mp4 output.mp4 --preset social
```

### Verification Profiles

```bash
# Strict: BER < 0.05 = pass, > 0.15 = fail
python src/cli.py verify video.mp4 --secret abc123 --profile strict

# Balanced: BER < 0.1 = pass, > 0.2 = fail (default)
python src/cli.py verify video.mp4 --secret abc123 --profile balanced

# Lenient: BER < 0.15 = pass, > 0.3 = fail
python src/cli.py verify video.mp4 --secret abc123 --profile lenient
```

## 📊 BER Analysis & Monitoring

### BER Timeline Visualization

```bash
# Export BER analysis artifacts
python src/cli.py verify video.mp4 --secret abc123 --export-artifacts
# Creates: out/ber.csv, out/ber.png
```

### Human-Readable Summaries

```bash
python src/cli.py verify video.mp4 --secret abc123 --summary
# Output: "Valid: Yes (avg BER 0.067). Potential edits at frames 45-52 (high BER run)"
```

### Prometheus Metrics

- `verify_requests_total` - Request counters by status
- `verify_latency_seconds` - Processing time histograms  
- `avg_ber` - BER distribution tracking

Access at `http://localhost:8000/metrics`

### Health Monitoring

```bash
# Health check
curl http://localhost:8000/healthz

# Readiness check (model loading)
curl http://localhost:8000/readyz
```

## 🐳 Docker Deployment

```bash
# Single container
docker build -t video-integrity-sdk .
docker run -p 8000:8000 -e API_KEY=your-key video-integrity-sdk

# Full monitoring stack
docker-compose up -d
# Access: API (8000), Prometheus (9090), Grafana (3000)
```

## 🧪 Development

```bash
# Setup development environment
make dev-setup

# Code quality
make format  # Black + ruff formatting
make lint    # Type checking with mypy
make test    # Full test suite (47 tests)
make check   # Quick validation

# Testing
make test-fast      # Skip slow tests
make test-unit      # Unit tests only  
make test-integration  # Integration tests
```

## 📚 Examples

- [`examples/embed_and_verify.py`](examples/embed_and_verify.py) - Complete workflow
- [`examples/verify_via_api.py`](examples/verify_via_api.py) - API integration
- [`js-sdk/examples/node-verify.js`](js-sdk/examples/node-verify.js) - Node.js usage

## 📖 Documentation

- [Implementation Summary](IMPLEMENTATION_SUMMARY.md) - Detailed feature overview
- [API Documentation](http://localhost:8000/docs) - Interactive OpenAPI docs
- [Design Rationale](docs/design.md) - Architecture decisions
- [Contributing Guide](CONTRIBUTING.md) - Development workflow

## 🔧 Troubleshooting

**FFmpeg not found?**
```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian  
sudo apt install ffmpeg

# Test without FFmpeg
python test_quick.py
```

**API authentication errors?**
```bash
export API_KEY=your-secret-key
# or set in config.yaml
```

**Import errors in examples?**
```bash
# Run from project root
cd video-integrity-sdk
python examples/embed_and_verify.py
```

---

**🎉 Ready for production use with comprehensive testing, monitoring, and developer experience!**
