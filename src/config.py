# config.py: Configuration management with presets and validation

import os
import yaml
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
from dataclasses import dataclass

class EmbedConfig(BaseModel):
    crf: int = Field(default=23, ge=0, le=51)
    secret: Optional[str] = None
    seed: Optional[int] = None
    preset: Optional[str] = None
    
class VerifyConfig(BaseModel):
    n_frames: int = Field(default=10, ge=1, le=100)
    profile: str = Field(default="balanced")
    sample_strategy: str = Field(default="uniform")
    export_artifacts: bool = Field(default=False)
    
class APIConfig(BaseModel):
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000, ge=1, le=65535)
    api_key: Optional[str] = None
    cors_origins: list = Field(default_factory=lambda: ["http://localhost:3000"])
    max_file_size_mb: int = Field(default=100, ge=1, le=1000)
    
class Config(BaseModel):
    embed: EmbedConfig = Field(default_factory=EmbedConfig)
    verify: VerifyConfig = Field(default_factory=VerifyConfig)
    api: APIConfig = Field(default_factory=APIConfig)

# Preset mappings
ATTACK_PRESETS = {
    "reencode_light": {"crf": 23, "preset": "medium"},
    "reencode_heavy": {"crf": 35, "preset": "slow"},
    "social": {"crf": 28, "preset": "fast", "scale": "720:-1"}
}

PROFILE_THRESHOLDS = {
    "strict": {"pass": 0.05, "fail": 0.15},
    "balanced": {"pass": 0.1, "fail": 0.2}, 
    "lenient": {"pass": 0.15, "fail": 0.3}
}

def load_config(config_path: Optional[str] = None) -> Config:
    """Load configuration from file or environment variables"""
    config_data = {}
    
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f) or {}
    
    # Override with environment variables (only if not already set in config)
    env_overrides = {
        "api": {
            "api_key": os.getenv("API_KEY"),
            "host": os.getenv("API_HOST"),
            "port": os.getenv("API_PORT"),
            "max_file_size_mb": os.getenv("MAX_FILE_SIZE_MB")
        }
    }
    
    # Merge configs (env vars override config file)
    for section, values in env_overrides.items():
        if section not in config_data:
            config_data[section] = {}
        for key, value in values.items():
            if value is not None:
                if key in ["port", "max_file_size_mb"]:
                    config_data[section][key] = int(value)
                else:
                    config_data[section][key] = value
    
    return Config(**config_data)

def get_attack_preset(preset_name: str) -> Dict[str, Any]:
    """Get FFmpeg parameters for attack preset"""
    return ATTACK_PRESETS.get(preset_name, {})

def get_profile_thresholds(profile: str) -> Dict[str, float]:
    """Get BER thresholds for verification profile"""
    return PROFILE_THRESHOLDS.get(profile, PROFILE_THRESHOLDS["balanced"])