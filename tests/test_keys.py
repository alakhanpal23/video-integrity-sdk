import os
import tempfile
import keys

# test_keys.py: Unit tests for key management (keys.py)
#
# - test_derive_secret: Checks HMAC-SHA256 derivation
# - test_store_and_get_secret: Checks SQLite storage/retrieval
#
# To add new key management tests, define new test_ functions below.

def test_derive_secret():
    video_id = "testvideo123"
    secret = keys.derive_secret(video_id)
    assert isinstance(secret, str)
    assert len(secret) >= 8  # 32 bits = 8 hex chars

def test_store_and_get_secret():
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        db_path = tmp.name
    os.environ["VIDEO_SDK_KEY_DB"] = db_path
    video_id = "vid1"
    secret = "deadbeef"
    keys.store_secret(video_id, secret)
    out = keys.get_secret(video_id)
    assert out == secret
    os.remove(db_path) 