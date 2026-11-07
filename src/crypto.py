"""At-rest protection (add-on A2): Fernet key derived from a passphrase with scrypt."""
from __future__ import annotations

import base64
import os

MAGIC = b"LRAG1"


def _fernet(passphrase: str, salt: bytes):
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives.kdf.scrypt import Scrypt

    kdf = Scrypt(salt=salt, length=32, n=2**14, r=8, p=1)
    return Fernet(base64.urlsafe_b64encode(kdf.derive(passphrase.encode("utf-8"))))


def encrypt_bytes(data: bytes, passphrase: str) -> bytes:
    salt = os.urandom(16)
    return MAGIC + salt + _fernet(passphrase, salt).encrypt(data)


def decrypt_bytes(blob: bytes, passphrase: str) -> bytes:
    if not blob.startswith(MAGIC):
        raise ValueError("Not an encrypted local-rag blob")
    salt = blob[len(MAGIC): len(MAGIC) + 16]
    return _fernet(passphrase, salt).decrypt(blob[len(MAGIC) + 16:])


def get_passphrase(prompt: str = "Passphrase: ") -> str:
    pw = os.environ.get("RAG_PASSPHRASE")
    if pw:
        return pw
    import getpass

    return getpass.getpass(prompt)
