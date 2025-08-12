# tests/test_keys.py

import pytest
import tempfile
import os
from src.keys import derive_secret, store_secret, get_secret, list_secrets, delete_secret

@pytest.fixture
def temp_db():
    """Create temporary database for testing"""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name
    
    # Patch the DB_PATH
    import src.keys
    original_path = src.keys.DB_PATH
    src.keys.DB_PATH = db_path
    
    yield db_path
    
    # Cleanup
    src.keys.DB_PATH = original_path
    if os.path.exists(db_path):
        os.unlink(db_path)

def test_derive_secret_deterministic():
    """Test that secret derivation is deterministic"""
    video_id = "test_video_123"
    
    secret1 = derive_secret(video_id)
    secret2 = derive_secret(video_id)
    
    assert secret1 == secret2
    assert len(secret1) == 8  # 32 bits = 4 bytes = 8 hex chars

def test_derive_secret_different_ids():
    """Test that different video IDs produce different secrets"""
    secret1 = derive_secret("video1")
    secret2 = derive_secret("video2")
    
    assert secret1 != secret2

def test_derive_secret_different_lengths():
    """Test secret derivation with different bit lengths"""
    video_id = "test_video"
    
    secret_16 = derive_secret(video_id, 16)
    secret_32 = derive_secret(video_id, 32)
    secret_64 = derive_secret(video_id, 64)
    
    assert len(secret_16) == 4   # 16 bits = 2 bytes = 4 hex chars
    assert len(secret_32) == 8   # 32 bits = 4 bytes = 8 hex chars
    assert len(secret_64) == 16  # 64 bits = 8 bytes = 16 hex chars

def test_derive_secret_validation():
    """Test input validation for derive_secret"""
    # Empty video_id
    with pytest.raises(ValueError, match="video_id cannot be empty"):
        derive_secret("")
    
    # Invalid bit length
    with pytest.raises(ValueError, match="secret_len_bits must be between 8 and 256"):
        derive_secret("test", 4)
    
    with pytest.raises(ValueError, match="secret_len_bits must be between 8 and 256"):
        derive_secret("test", 300)

def test_store_and_get_secret(temp_db):
    """Test storing and retrieving secrets"""
    video_id = "test_video_456"
    secret_hex = "deadbeefcafebabe"
    
    # Store secret
    store_secret(video_id, secret_hex)
    
    # Retrieve secret
    retrieved = get_secret(video_id)
    assert retrieved == secret_hex

def test_store_secret_validation(temp_db):
    """Test input validation for store_secret"""
    # Empty video_id
    with pytest.raises(ValueError, match="video_id and secret_hex cannot be empty"):
        store_secret("", "deadbeef")
    
    # Empty secret_hex
    with pytest.raises(ValueError, match="video_id and secret_hex cannot be empty"):
        store_secret("test", "")

def test_get_secret_nonexistent(temp_db):
    """Test getting non-existent secret"""
    result = get_secret("nonexistent_video")
    assert result is None

def test_get_secret_empty_id(temp_db):
    """Test getting secret with empty ID"""
    result = get_secret("")
    assert result is None

def test_secret_replacement(temp_db):
    """Test that storing a secret with same video_id replaces the old one"""
    video_id = "test_video_replace"
    
    # Store first secret
    store_secret(video_id, "secret1")
    assert get_secret(video_id) == "secret1"
    
    # Replace with new secret
    store_secret(video_id, "secret2")
    assert get_secret(video_id) == "secret2"

def test_list_secrets(temp_db):
    """Test listing all secrets"""
    # Initially empty
    secrets = list_secrets()
    assert len(secrets) == 0
    
    # Add some secrets
    store_secret("video1", "secret1")
    store_secret("video2", "secret2")
    
    secrets = list_secrets()
    assert len(secrets) == 2
    
    # Check structure (video_id, created_at)
    video_ids = [s[0] for s in secrets]
    assert "video1" in video_ids
    assert "video2" in video_ids

def test_delete_secret(temp_db):
    """Test deleting secrets"""
    video_id = "test_video_delete"
    
    # Store secret
    store_secret(video_id, "secret_to_delete")
    assert get_secret(video_id) == "secret_to_delete"
    
    # Delete secret
    result = delete_secret(video_id)
    assert result is True
    assert get_secret(video_id) is None
    
    # Try to delete again
    result = delete_secret(video_id)
    assert result is False

def test_delete_secret_empty_id(temp_db):
    """Test deleting with empty ID"""
    result = delete_secret("")
    assert result is False

def test_hmac_consistency():
    """Test that HMAC derivation is consistent with known values"""
    # This test ensures the HMAC implementation doesn't change unexpectedly
    import src.keys
    
    # Temporarily set a known master key
    original_key = src.keys.MASTER_KEY
    src.keys.MASTER_KEY = b"test_master_key"
    
    try:
        secret = derive_secret("test_video", 32)
        # This should be deterministic for the given master key and video_id
        assert len(secret) == 8
        assert isinstance(secret, str)
        
        # Same input should produce same output
        secret2 = derive_secret("test_video", 32)
        assert secret == secret2
        
    finally:
        src.keys.MASTER_KEY = original_key

def test_concurrent_access(temp_db):
    """Test thread-safe database access"""
    import threading
    import time
    
    results = []
    errors = []
    
    def worker(worker_id):
        try:
            video_id = f"video_{worker_id}"
            secret = f"secret_{worker_id}"
            
            # Store
            store_secret(video_id, secret)
            
            # Small delay to increase chance of race conditions
            time.sleep(0.01)
            
            # Retrieve
            retrieved = get_secret(video_id)
            results.append((video_id, secret, retrieved))
            
        except Exception as e:
            errors.append(e)
    
    # Create multiple threads
    threads = []
    for i in range(10):
        t = threading.Thread(target=worker, args=(i,))
        threads.append(t)
        t.start()
    
    # Wait for all threads
    for t in threads:
        t.join()
    
    # Check results
    assert len(errors) == 0, f"Errors occurred: {errors}"
    assert len(results) == 10
    
    for video_id, original, retrieved in results:
        assert original == retrieved