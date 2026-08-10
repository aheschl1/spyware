"""The entry point into blob storage, the counterpart of ``DatabasePipe``.

    async with BlobPipe() as blobs:
        info = await blobs.put(key, data, content_type="audio/wav")
        url = await blobs.presign_get(key)

Object writes are durable as soon as they land; there is no transaction to roll
back. :mod:`services.audio` cleans up objects whose database row failed.
"""

from types import TracebackType

import aioboto3
from botocore.config import Config

from storage.config import StorageSettings, get_settings
from storage.s3 import S3BlobStore

_session = aioboto3.Session()

# Signature v4 explicitly: against a custom endpoint botocore can otherwise fall
# back to the v2 presigning scheme, which AWS S3 no longer accepts in regions
# created after 2014 -- so presigned URLs would break on a move off MinIO.
_CONFIG = Config(signature_version="s3v4")


class BlobPipe:
    def __init__(self, *, settings: StorageSettings | None = None) -> None:
        self._settings = settings or get_settings()
        self._cm = None

    async def __aenter__(self) -> S3BlobStore:
        settings = self._settings
        self._cm = _session.client(
            "s3",
            endpoint_url=settings.endpoint_url,
            aws_access_key_id=settings.access_key,
            aws_secret_access_key=settings.secret_key,
            region_name=settings.region,
            config=_CONFIG,
        )
        client = await self._cm.__aenter__()
        return S3BlobStore(client, settings.bucket, settings.presign_expiry_seconds)

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool | None:
        cm, self._cm = self._cm, None
        if cm is None:
            return None
        return await cm.__aexit__(exc_type, exc, tb)
