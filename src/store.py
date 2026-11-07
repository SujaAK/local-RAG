"""SQLite storage + blob helpers, with optional encryption at rest (decrypt into memory)."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from . import crypto
from .utils import rpath

SCHEMA = """
CREATE TABLE IF NOT EXISTS papers(
  paper_id TEXT PRIMARY KEY, sha256 TEXT, filename TEXT, title TEXT, authors TEXT,
  year INTEGER, venue TEXT, n_pages INTEGER);
CREATE TABLE IF NOT EXISTS sections(
  paper_id TEXT, idx INTEGER, name TEXT, text TEXT, page_marks TEXT,
  PRIMARY KEY(paper_id, idx));
CREATE TABLE IF NOT EXISTS chunks(
  chunk_id TEXT PRIMARY KEY, paper_id TEXT, section TEXT, page INTEGER,
  start INTEGER, end INTEGER, text TEXT);
CREATE TABLE IF NOT EXISTS hyperedges(
  edge_id INTEGER PRIMARY KEY AUTOINCREMENT, paper_id TEXT, chunk_id TEXT,
  method TEXT, task TEXT, dataset TEXT, metric TEXT, value TEXT, condition TEXT,
  limitation TEXT, span TEXT);
"""


def _plain(cfg) -> Path:
    return rpath(cfg, "db")


def _enc(cfg) -> Path:
    p = _plain(cfg)
    return p.with_name(p.name + ".enc")


def open_db(cfg, passphrase: str | None = None) -> sqlite3.Connection:
    p = _plain(cfg)
    p.parent.mkdir(parents=True, exist_ok=True)
    if cfg.privacy.encrypt_at_rest:
        conn = sqlite3.connect(":memory:")
        if _enc(cfg).exists():
            conn.deserialize(crypto.decrypt_bytes(_enc(cfg).read_bytes(), passphrase))
        elif p.exists():
            disk = sqlite3.connect(p)
            disk.backup(conn)
            disk.close()
    else:
        conn = sqlite3.connect(p)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


def close_db(conn: sqlite3.Connection, cfg, passphrase: str | None = None) -> None:
    conn.commit()
    if cfg.privacy.encrypt_at_rest:
        _enc(cfg).write_bytes(crypto.encrypt_bytes(conn.serialize(), passphrase))
        if _plain(cfg).exists():
            _plain(cfg).unlink()
    conn.close()


def write_blob(path: Path, data: bytes, cfg, passphrase: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if cfg.privacy.encrypt_at_rest:
        path.with_name(path.name + ".enc").write_bytes(crypto.encrypt_bytes(data, passphrase))
        if path.exists():
            path.unlink()
    else:
        path.write_bytes(data)


def read_blob(path: Path, cfg, passphrase: str | None = None) -> bytes:
    enc = path.with_name(path.name + ".enc")
    if enc.exists():
        return crypto.decrypt_bytes(enc.read_bytes(), passphrase)
    return path.read_bytes()


def load_chunks(conn) -> dict[str, dict]:
    rows = conn.execute(
        "SELECT c.*, p.title, p.year, p.authors FROM chunks c JOIN papers p USING(paper_id) "
        "ORDER BY c.chunk_id").fetchall()
    return {r["chunk_id"]: dict(r) for r in rows}


def dump_json(obj) -> bytes:
    return json.dumps(obj).encode("utf-8")


def write_text_private(path: Path, text: str, cfg, passphrase: str | None = None) -> Path:
    """Write text; encrypted (<name>.enc, plaintext never written) when privacy.encrypt_at_rest is on."""
    write_blob(path, text.encode("utf-8"), cfg, passphrase)
    return path.with_name(path.name + ".enc") if cfg.privacy.encrypt_at_rest else path


def read_text_private(path: Path, passphrase: str | None = None) -> str:
    if path.suffix == ".enc":
        return crypto.decrypt_bytes(path.read_bytes(), passphrase).decode("utf-8")
    return path.read_text(encoding="utf-8")
