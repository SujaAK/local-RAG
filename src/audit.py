"""B4 / T11. Privacy audit helpers: offline guard, environment checks, socket listing, encryption timing."""
from __future__ import annotations

import os
import socket
import time

LOCAL = {"127.0.0.1", "::1", "localhost"}


def block_network() -> None:
    """In-process guard: any non-localhost connection raises. Complements (does not replace) the
    OS-level test in scripts/offline_run.sh, which is the T11 pass criterion."""
    real_connect = socket.socket.connect

    def guarded(self, address):
        if self.family not in (socket.AF_INET, socket.AF_INET6):  # Unix sockets etc. are local
            return real_connect(self, address)
        host = address[0] if isinstance(address, tuple) else address
        if isinstance(host, str) and host not in LOCAL and not host.startswith("127."):
            raise ConnectionError(f"Outbound connection blocked by local-rag offline guard: {host}")
        return real_connect(self, address)

    socket.socket.connect = guarded  # type: ignore[assignment]


def check_env() -> list[dict]:
    host = os.environ.get("OLLAMA_HOST", "127.0.0.1")
    checks = [
        ("HF_HUB_OFFLINE=1", os.environ.get("HF_HUB_OFFLINE") == "1"),
        ("TRANSFORMERS_OFFLINE=1", os.environ.get("TRANSFORMERS_OFFLINE") == "1"),
        ("OLLAMA_HOST is localhost", host.split(":")[0] in LOCAL or host.startswith("127.")),
        ("Streamlit telemetry off", _streamlit_telemetry_off()),
    ]
    return [{"check": n, "ok": bool(ok)} for n, ok in checks]


def _streamlit_telemetry_off() -> bool:
    from .utils import ROOT

    p = ROOT / ".streamlit" / "config.toml"
    return p.exists() and "gatherUsageStats = false" in p.read_text()


def open_sockets() -> list[dict]:
    """Processes with non-local established connections (T11: list processes with open sockets)."""
    import psutil

    rows = []
    try:
        for c in psutil.net_connections(kind="inet"):
            if c.raddr and c.raddr.ip not in LOCAL and not c.raddr.ip.startswith("127."):
                rows.append({"pid": c.pid, "remote": f"{c.raddr.ip}:{c.raddr.port}", "status": c.status})
    except psutil.AccessDenied:
        rows.append({"error": "access denied; re-run with sufficient privileges"})
    return rows


def telemetry_packages() -> list[str]:
    import importlib.metadata as md

    flags = ("sentry", "posthog", "mixpanel", "segment", "analytics", "telemetry", "amplitude")
    return sorted({d.metadata["Name"] for d in md.distributions() if d.metadata["Name"]
                   and any(f in d.metadata["Name"].lower() for f in flags)})


def time_encryption(cfg, passphrase: str, size_mb: int = 16) -> dict:
    """Measure encrypt/decrypt overhead on a blob of the given size (report in T11 table)."""
    from . import crypto

    blob = os.urandom(size_mb * 2**20)
    t = time.perf_counter()
    enc = crypto.encrypt_bytes(blob, passphrase)
    t_enc = time.perf_counter() - t
    t = time.perf_counter()
    crypto.decrypt_bytes(enc, passphrase)
    return {"size_mb": size_mb, "encrypt_s": round(t_enc, 3), "decrypt_s": round(time.perf_counter() - t, 3)}


def run_audit(cfg=None) -> dict:
    return {"env": check_env(), "non_local_sockets": open_sockets(), "telemetry_like_packages": telemetry_packages()}
