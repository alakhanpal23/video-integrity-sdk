# tests/test_models.py

import pytest
import torch
import numpy as np
from src.encoder import WatermarkEncoder
from src.decoder import WatermarkDecoder

def test_encoder_initialization():
    """Test encoder model initialization"""
    encoder = WatermarkEncoder(secret_len=32)
    assert encoder.conv1.in_channels == 3
    assert encoder.conv1.out_channels == 16
    assert encoder.conv2.in_channels == 16
    assert encoder.conv2.out_channels == 3
    assert encoder.secret_fc.in_features == 32
    assert encoder.secret_fc.out_features == 3 * 64 * 64

def test_encoder_forward():
    """Test encoder forward pass"""
    encoder = WatermarkEncoder(secret_len=32)
    encoder.eval()
    
    # Create test inputs
    batch_size = 2
    frame = torch.randn(batch_size, 3, 256, 256)
    secret = torch.randn(batch_size, 32)
    
    # Forward pass
    with torch.no_grad():
        output = encoder(frame, secret)
    
    # Check output shape
    assert output.shape == frame.shape
    assert output.dtype == torch.float32

def test_encoder_different_sizes():
    """Test encoder with different input sizes"""
    encoder = WatermarkEncoder(secret_len=16)
    encoder.eval()
    
    sizes = [(128, 128), (256, 256), (512, 512)]
    
    for h, w in sizes:
        frame = torch.randn(1, 3, h, w)
        secret = torch.randn(1, 16)
        
        with torch.no_grad():
            output = encoder(frame, secret)
        
        assert output.shape == (1, 3, h, w)

def test_decoder_initialization():
    """Test decoder model initialization"""
    decoder = WatermarkDecoder(secret_len=32)
    assert decoder.conv1.in_channels == 3
    assert decoder.conv1.out_channels == 16
    assert decoder.conv2.in_channels == 16
    assert decoder.conv2.out_channels == 3
    assert decoder.fc.in_features == 3 * 64 * 64
    assert decoder.fc.out_features == 32

def test_decoder_forward():
    """Test decoder forward pass"""
    decoder = WatermarkDecoder(secret_len=32)
    decoder.eval()
    
    # Create test input
    batch_size = 2
    frame = torch.randn(batch_size, 3, 256, 256)
    
    # Forward pass
    with torch.no_grad():
        logits = decoder(frame)
    
    # Check output shape
    assert logits.shape == (batch_size, 32)
    assert logits.dtype == torch.float32

def test_decoder_decode_bits():
    """Test bit decoding from logits"""
    # Create test logits
    logits = torch.tensor([
        [2.0, -1.0, 0.5, -0.5],  # Should decode to [1, 0, 1, 0]
        [-2.0, 1.0, -0.5, 0.5]   # Should decode to [0, 1, 0, 1]
    ])
    
    bits = WatermarkDecoder.decode_bits(logits, threshold=0.5)
    
    expected = np.array([
        [1, 0, 1, 0],
        [0, 1, 0, 1]
    ], dtype=np.uint8)
    
    np.testing.assert_array_equal(bits, expected)

def test_decoder_decode_bits_threshold():
    """Test bit decoding with different thresholds"""
    logits = torch.tensor([[0.0, 1.0, -1.0, 0.6]])  # After sigmoid: [0.5, 0.73, 0.27, 0.65]
    
    # Threshold 0.5
    bits_05 = WatermarkDecoder.decode_bits(logits, threshold=0.5)
    expected_05 = np.array([[0, 1, 0, 1]], dtype=np.uint8)
    np.testing.assert_array_equal(bits_05, expected_05)
    
    # Threshold 0.7
    bits_07 = WatermarkDecoder.decode_bits(logits, threshold=0.7)
    expected_07 = np.array([[0, 1, 0, 0]], dtype=np.uint8)
    np.testing.assert_array_equal(bits_07, expected_07)

def test_encoder_decoder_consistency():
    """Test that encoder and decoder have consistent secret lengths"""
    secret_len = 24
    encoder = WatermarkEncoder(secret_len=secret_len)
    decoder = WatermarkDecoder(secret_len=secret_len)
    
    frame = torch.randn(1, 3, 256, 256)
    secret = torch.randn(1, secret_len)
    
    encoder.eval()
    decoder.eval()
    
    with torch.no_grad():
        # Encode
        watermarked = encoder(frame, secret)
        
        # Decode
        logits = decoder(watermarked)
        
        assert logits.shape == (1, secret_len)

def test_encoder_watermark_strength():
    """Test that watermark is subtle (small changes to original)"""
    encoder = WatermarkEncoder(secret_len=32)
    encoder.eval()
    
    frame = torch.randn(1, 3, 256, 256)
    secret = torch.randn(1, 32)
    
    with torch.no_grad():
        watermarked = encoder(frame, secret)
        
        # Watermark should be subtle
        diff = torch.abs(watermarked - frame)
        max_diff = torch.max(diff)
        mean_diff = torch.mean(diff)
        
        # These thresholds are based on the model architecture
        assert max_diff < 1.0, f"Maximum difference too large: {max_diff}"
        assert mean_diff < 0.1, f"Mean difference too large: {mean_diff}"

def test_model_gradients():
    """Test that models can compute gradients"""
    encoder = WatermarkEncoder(secret_len=16)
    decoder = WatermarkDecoder(secret_len=16)
    
    frame = torch.randn(1, 3, 128, 128, requires_grad=True)
    secret = torch.randn(1, 16, requires_grad=True)
    
    # Forward pass
    watermarked = encoder(frame, secret)
    logits = decoder(watermarked)
    
    # Compute loss
    target = torch.randint(0, 2, (1, 16), dtype=torch.float32)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
    
    # Backward pass
    loss.backward()
    
    # Check that gradients exist
    assert frame.grad is not None
    assert secret.grad is not None
    
    # Check encoder gradients
    for param in encoder.parameters():
        assert param.grad is not None
    
    # Check decoder gradients
    for param in decoder.parameters():
        assert param.grad is not None

def test_model_device_compatibility():
    """Test model device compatibility"""
    encoder = WatermarkEncoder(secret_len=32)
    decoder = WatermarkDecoder(secret_len=32)
    
    # Test CPU
    frame_cpu = torch.randn(1, 3, 256, 256)
    secret_cpu = torch.randn(1, 32)
    
    encoder.eval()
    decoder.eval()
    
    with torch.no_grad():
        watermarked_cpu = encoder(frame_cpu, secret_cpu)
        logits_cpu = decoder(watermarked_cpu)
    
    assert watermarked_cpu.device.type == 'cpu'
    assert logits_cpu.device.type == 'cpu'
    
    # Test CUDA if available
    if torch.cuda.is_available():
        encoder_cuda = encoder.cuda()
        decoder_cuda = decoder.cuda()
        frame_cuda = frame_cpu.cuda()
        secret_cuda = secret_cpu.cuda()
        
        with torch.no_grad():
            watermarked_cuda = encoder_cuda(frame_cuda, secret_cuda)
            logits_cuda = decoder_cuda(watermarked_cuda)
        
        assert watermarked_cuda.device.type == 'cuda'
        assert logits_cuda.device.type == 'cuda'

def test_model_deterministic():
    """Test that models are deterministic with same inputs"""
    torch.manual_seed(42)
    encoder1 = WatermarkEncoder(secret_len=32)
    decoder1 = WatermarkDecoder(secret_len=32)
    
    torch.manual_seed(42)
    encoder2 = WatermarkEncoder(secret_len=32)
    decoder2 = WatermarkDecoder(secret_len=32)
    
    # Models should have same initial weights
    for p1, p2 in zip(encoder1.parameters(), encoder2.parameters()):
        torch.testing.assert_close(p1, p2)
    
    for p1, p2 in zip(decoder1.parameters(), decoder2.parameters()):
        torch.testing.assert_close(p1, p2)
    
    # Same inputs should produce same outputs
    torch.manual_seed(123)
    frame = torch.randn(1, 3, 256, 256)
    secret = torch.randn(1, 32)
    
    encoder1.eval()
    encoder2.eval()
    decoder1.eval()
    decoder2.eval()
    
    with torch.no_grad():
        out1 = encoder1(frame, secret)
        out2 = encoder2(frame, secret)
        torch.testing.assert_close(out1, out2)
        
        logits1 = decoder1(frame)
        logits2 = decoder2(frame)
        torch.testing.assert_close(logits1, logits2)

def test_model_parameter_count():
    """Test that models have reasonable parameter counts"""
    encoder = WatermarkEncoder(secret_len=32)
    decoder = WatermarkDecoder(secret_len=32)
    
    encoder_params = sum(p.numel() for p in encoder.parameters())
    decoder_params = sum(p.numel() for p in decoder.parameters())
    
    # These are rough estimates based on the architecture
    assert 10000 < encoder_params < 1000000, f"Encoder params: {encoder_params}"
    assert 10000 < decoder_params < 1000000, f"Decoder params: {decoder_params}"
    
    print(f"Encoder parameters: {encoder_params:,}")
    print(f"Decoder parameters: {decoder_params:,}")

def test_batch_processing():
    """Test models with different batch sizes"""
    encoder = WatermarkEncoder(secret_len=32)
    decoder = WatermarkDecoder(secret_len=32)
    
    encoder.eval()
    decoder.eval()
    
    batch_sizes = [1, 2, 4, 8]
    
    for batch_size in batch_sizes:
        frame = torch.randn(batch_size, 3, 256, 256)
        secret = torch.randn(batch_size, 32)
        
        with torch.no_grad():
            watermarked = encoder(frame, secret)
            logits = decoder(watermarked)
        
        assert watermarked.shape == (batch_size, 3, 256, 256)
        assert logits.shape == (batch_size, 32)