"""B4 / T11. Privacy audit helpers: offline guard, environment checks, socket listing, encryption timing."""
from __future__ import annotations

import ipaddress
import os
import socket
import time
from urllib.parse import urlparse

LOCAL = {"127.0.0.1", "::1", "localhost"}


def is_local_host(host) -> bool:
    """True for loopback only: localhost, 127.0.0.0/8, ::1 and IPv4-mapped loopback (::ffff:127.0.0.1)."""
    if not isinstance(host, str):
        return False
    h = host.strip().strip("[]").split("%")[0].lower()
    if h in LOCAL:
        return True
    try:
        ip = ipaddress.ip_address(h)
    except ValueError:
        return False
    if ip.version == 6 and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    return ip.is_loopback


def host_of(value: str) -> str:
    """Host part of an address such as '127.0.0.1:11434', 'http://localhost:11434' or '[::1]:11434'."""
    v = (value or "").strip()
    if "://" in v:
        v = urlparse(v).netloc
    if v.startswith("["):
        return v[1:v.find("]")] if "]" in v else v
    if v.count(":") == 1:
        v = v.split(":")[0]
    return v or "127.0.0.1"


def block_network() -> None:
    """In-process guard: any non-loopback connection raises (connect and connect_ex). Safe to call repeatedly
    (Streamlit reruns the script). Complements, does not replace, the OS-level test in scripts/run_offline.sh,
    which is the T11 pass criterion."""
    if getattr(socket.socket.connect, "_rag_guard", False):
        return
    real_connect, real_connect_ex = socket.socket.connect, socket.socket.connect_ex

    def _check(self, address):
        if self.family not in (socket.AF_INET, socket.AF_INET6):  # Unix sockets etc. are local
            return
        host = address[0] if isinstance(address, tuple) else address
        if isinstance(host, str) and not is_local_host(host):
            raise ConnectionError(f"Outbound connection blocked by local-rag offline guard: {host}")

    def guarded(self, address):
        _check(self, address)
        return real_connect(self, address)

    def guarded_ex(self, address):
        _check(self, address)
        return real_connect_ex(self, address)

    guarded._rag_guard = True  # type: ignore[attr-defined]
    guarded_ex._rag_guard = True  # type: ignore[attr-defined]
    socket.socket.connect = guarded  # type: ignore[assignment]
    socket.socket.connect_ex = guarded_ex  # type: ignore[assignment]


def check_env() -> list[dict]:
    host = os.environ.get("OLLAMA_HOST", "127.0.0.1")
    checks = [
        ("HF_HUB_OFFLINE=1", os.environ.get("HF_HUB_OFFLINE") == "1"),
        ("TRANSFORMERS_OFFLINE=1", os.environ.get("TRANSFORMERS_OFFLINE") == "1"),
        ("OLLAMA_HOST is localhost", is_local_host(host_of(host))),
        ("Streamlit telemetry off", _streamlit_telemetry_off()),
    ]
    return [{"check": n, "ok": bool(ok)} for n, ok in checks]


def _streamlit_telemetry_off() -> bool:
    from .utils import ROOT

    p = ROOT / ".streamlit" / "config.toml"
    return p.exists() and "gatherUsageStats = false" in p.read_text(encoding="utf-8")


def open_sockets() -> list[dict]:
    """Processes with non-local established connections (T11: list processes with open sockets)."""
    import psutil

    rows = []
    try:
        for c in psutil.net_connections(kind="inet"):
            if c.raddr and not is_local_host(c.raddr.ip):
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