# tests/test_config.py

import pytest
import tempfile
import os
import yaml
from src.config import load_config, get_attack_preset, get_profile_thresholds, Config

def test_default_config():
    """Test default configuration loading"""
    config = load_config()
    assert isinstance(config, Config)
    assert config.embed.crf == 23
    assert config.verify.n_frames == 10
    assert config.api.port == 8000

def test_config_from_file():
    """Test loading configuration from YAML file"""
    config_data = {
        "embed": {"crf": 20, "seed": 42},
        "verify": {"n_frames": 15, "profile": "strict"},
        "api": {"port": 9000, "max_file_size_mb": 200}
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config_data, f)
        config_path = f.name
    
    try:
        config = load_config(config_path)
        assert config.embed.crf == 20
        assert config.embed.seed == 42
        assert config.verify.n_frames == 15
        assert config.api.port == 9000
        assert config.api.max_file_size_mb == 200
    finally:
        os.unlink(config_path)

def test_env_override():
    """Test environment variable overrides"""
    os.environ["API_KEY"] = "test-key-123"
    os.environ["API_PORT"] = "7000"
    
    try:
        config = load_config()
        assert config.api.api_key == "test-key-123"
        assert config.api.port == 7000
    finally:
        del os.environ["API_KEY"]
        del os.environ["API_PORT"]

def test_attack_presets():
    """Test attack preset mappings"""
    light = get_attack_preset("reencode_light")
    assert light["crf"] == 23
    assert light["preset"] == "medium"
    
    heavy = get_attack_preset("reencode_heavy")
    assert heavy["crf"] == 35
    
    social = get_attack_preset("social")
    assert social["crf"] == 28
    assert "scale" in social
    
    # Test unknown preset
    unknown = get_attack_preset("unknown")
    assert unknown == {}

def test_profile_thresholds():
    """Test BER threshold profiles"""
    strict = get_profile_thresholds("strict")
    assert strict["pass"] == 0.05
    assert strict["fail"] == 0.15
    
    balanced = get_profile_thresholds("balanced")
    assert balanced["pass"] == 0.1
    assert balanced["fail"] == 0.2
    
    lenient = get_profile_thresholds("lenient")
    assert lenient["pass"] == 0.15
    assert lenient["fail"] == 0.3
    
    # Test unknown profile defaults to balanced
    unknown = get_profile_thresholds("unknown")
    assert unknown == balanced

def test_config_validation():
    """Test pydantic validation"""
    # Test invalid CRF
    with pytest.raises(ValueError):
        Config(embed={"crf": 100})  # CRF must be 0-51
    
    # Test invalid port
    with pytest.raises(ValueError):
        Config(api={"port": 70000})  # Port must be 1-65535