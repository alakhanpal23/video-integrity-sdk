import os
import hmac
import hashlib
import sqlite3
from typing import Optional

# keys.py: Key management for watermark secrets
#
# - derive_secret: HMAC-SHA256 per-video secret from master key (env: VIDEO_SDK_MASTER_KEY)
# - store_secret/get_secret: Store and retrieve secrets in SQLite DB (env: VIDEO_SDK_KEY_DB)
#
# To change key derivation, edit derive_secret().
# To use a different DB or add encryption, update get_db().
#
# Used by encoder.py and api.py for secret management.

DB_PATH = os.environ.get("VIDEO_SDK_KEY_DB", "video_secrets.db")
MASTER_KEY = os.environ.get("VIDEO_SDK_MASTER_KEY", "default_master_key").encode()

# --- SQLite helpers ---
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
    CREATE TABLE IF NOT EXISTS secrets (
        video_id TEXT PRIMARY KEY,
        secret_hex TEXT NOT NULL
    )
    """)
    return conn

def store_secret(video_id: str, secret_hex: str):
    conn = get_db()
    with conn:
        conn.execute("REPLACE INTO secrets (video_id, secret_hex) VALUES (?, ?)", (video_id, secret_hex))
    conn.close()

def get_secret(video_id: str) -> Optional[str]:
    conn = get_db()
    cur = conn.execute("SELECT secret_hex FROM secrets WHERE video_id = ?", (video_id,))
    row = cur.fetchone()
    conn.close()
    return row[0] if row else None

# --- HMAC-SHA256 derivation ---
def derive_secret(video_id: str, secret_len_bits: int = 32) -> str:
    h = hmac.new(MASTER_KEY, video_id.encode(), hashlib.sha256).digest()
    # Use first N bits as secret
    n_bytes = (secret_len_bits + 7) // 8
    secret_bytes = h[:n_bytes]
    return secret_bytes.hex() 