"""Password hashing and API-token generation."""

import hashlib
import secrets
from functools import lru_cache

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError

_hasher = PasswordHasher()

TOKEN_BYTES = 32


def hash_password(password: str) -> str:
    return _hasher.hash(password)


def verify_password(password_hash: str, password: str) -> bool:
    try:
        return _hasher.verify(password_hash, password)
    except (VerifyMismatchError, VerificationError, InvalidHashError):
        return False


def needs_rehash(password_hash: str) -> bool:
    """True when the hash was made with outdated argon2 parameters."""
    try:
        return _hasher.check_needs_rehash(password_hash)
    except InvalidHashError:
        return True


@lru_cache(maxsize=1)
def _dummy_hash() -> str:
    return _hasher.hash("audio-pipeline-dummy-password")


def waste_password_time() -> None:
    """Verify against a throwaway hash.

    Called when no user matched, so an unknown email costs about the same as a
    known one and account existence does not leak through response timing.
    """
    verify_password(_dummy_hash(), "not-the-password")


def generate_token() -> str:
    """A high-entropy, URL-safe API token."""
    return secrets.token_urlsafe(TOKEN_BYTES)


def hash_token(token: str) -> bytes:
    """Digest stored in ``auth_tokens.token_hash``.

    Plain SHA-256: tokens are high-entropy random secrets, so they are not
    brute-forceable from the digest, and this runs on every request.
    """
    return hashlib.sha256(token.encode("utf-8")).digest()
