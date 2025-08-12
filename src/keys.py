import os
import hmac
import hashlib
import sqlite3
from typing import Optional
import threading

# Thread-safe database access
_db_lock = threading.Lock()

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
    with _db_lock:
        conn = sqlite3.connect(DB_PATH)
        conn.execute("""
        CREATE TABLE IF NOT EXISTS secrets (
            video_id TEXT PRIMARY KEY,
            secret_hex TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        return conn

def store_secret(video_id: str, secret_hex: str):
    if not video_id or not secret_hex:
        raise ValueError("video_id and secret_hex cannot be empty")
    
    conn = get_db()
    try:
        with conn:
            conn.execute("REPLACE INTO secrets (video_id, secret_hex) VALUES (?, ?)", (video_id, secret_hex))
    finally:
        conn.close()

def get_secret(video_id: str) -> Optional[str]:
    if not video_id:
        return None
        
    conn = get_db()
    try:
        cur = conn.execute("SELECT secret_hex FROM secrets WHERE video_id = ?", (video_id,))
        row = cur.fetchone()
        return row[0] if row else None
    finally:
        conn.close()

# --- HMAC-SHA256 derivation ---
def derive_secret(video_id: str, secret_len_bits: int = 32) -> str:
    if not video_id:
        raise ValueError("video_id cannot be empty")
    if not (8 <= secret_len_bits <= 256):
        raise ValueError("secret_len_bits must be between 8 and 256")
        
    h = hmac.new(MASTER_KEY, video_id.encode(), hashlib.sha256).digest()
    # Use first N bits as secret
    n_bytes = (secret_len_bits + 7) // 8
    secret_bytes = h[:n_bytes]
    return secret_bytes.hex()

def list_secrets() -> list:
    """List all stored secrets (for testing/admin)"""
    conn = get_db()
    try:
        cur = conn.execute("SELECT video_id, created_at FROM secrets ORDER BY created_at DESC")
        return cur.fetchall()
    finally:
        conn.close()

def delete_secret(video_id: str) -> bool:
    """Delete a secret by video_id"""
    if not video_id:
        return False
        
    conn = get_db()
    try:
        with conn:
            cur = conn.execute("DELETE FROM secrets WHERE video_id = ?", (video_id,))
            return cur.rowcount > 0
    finally:
        conn.close()