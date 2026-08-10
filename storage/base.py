"""The interface every blob backend implements.

Code outside ``storage/`` depends on :class:`BlobStore`, not on a concrete
backend. The only implementation is :class:`~storage.s3.S3BlobStore`.
"""

from collections.abc import AsyncIterator
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict


class BlobInfo(BaseModel):
    """What the store recorded for an object."""

    model_config = ConfigDict(frozen=True)

    bucket: str
    key: str
    byte_size: int
    content_type: str


class BlobNotFoundError(Exception):
    """No object exists at that key."""

    def __init__(self, key: str) -> None:
        super().__init__(f"no object at key {key!r}")
        self.key = key


@runtime_checkable
class BlobStore(Protocol):
    async def ensure_bucket(self) -> None:
        """Create the target bucket if it does not already exist."""
        ...

    async def put(self, key: str, data: bytes, content_type: str) -> BlobInfo: ...

    async def get(self, key: str) -> bytes: ...

    def stream(
        self,
        key: str,
        chunk_size: int = 1 << 20,
        start: int | None = None,
        end: int | None = None,
    ) -> AsyncIterator[bytes]:
        """Yield the object in chunks, optionally only the inclusive byte range."""
        ...

    async def exists(self, key: str) -> bool: ...

    async def delete(self, key: str) -> bool:
        """Delete one object. ``False`` if it was not there."""
        ...

    async def delete_prefix(self, prefix: str) -> int:
        """Delete every object under a prefix. Returns the count."""
        ...

    async def presign_get(self, key: str, expires_in: int | None = None) -> str: ...

    async def presign_put(
        self,
        key: str,
        content_type: str | None = None,
        expires_in: int | None = None,
    ) -> str: ...
