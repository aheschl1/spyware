"""S3 implementation of :class:`~storage.base.BlobStore`.

The only module that imports an S3 client. Switching between MinIO, AWS S3, R2
and the like is a `.env` change.
"""

from collections.abc import AsyncIterator
from typing import Any

from botocore.exceptions import ClientError

from storage.base import BlobInfo, BlobNotFoundError

# Codes S3 uses for "it isn't there", across head_* and get_* calls.
_MISSING = {"404", "NoSuchKey", "NoSuchBucket", "NotFound"}

_DELETE_BATCH = 1000  # S3's maximum objects per delete_objects call


def _error_code(exc: ClientError) -> str:
    return str(exc.response.get("Error", {}).get("Code", ""))


class S3BlobStore:
    """Operates on one bucket through an already-open aioboto3 client."""

    def __init__(self, client: Any, bucket: str, presign_expiry_seconds: int = 3600) -> None:
        self._client = client
        self._bucket = bucket
        self._presign_expiry = presign_expiry_seconds

    @property
    def bucket(self) -> str:
        return self._bucket

    async def ensure_bucket(self) -> None:
        """Check the bucket is reachable, creating it if the credentials allow.

        On the shared store this application's credentials are scoped to its own
        bucket, so a missing bucket has to be provisioned by an operator.
        """
        try:
            await self._client.head_bucket(Bucket=self._bucket)
            return
        except ClientError as exc:
            if _error_code(exc) not in _MISSING:
                raise
        try:
            await self._client.create_bucket(Bucket=self._bucket)
        except ClientError as exc:
            if _error_code(exc) != "AccessDenied":
                raise
            raise PermissionError(
                f"bucket {self._bucket!r} does not exist and these credentials may not "
                f"create it. Provision it on the shared store first: "
                f"builds/minio_add_service.sh {self._bucket}"
            ) from exc

    async def put(self, key: str, data: bytes, content_type: str) -> BlobInfo:
        await self._client.put_object(
            Bucket=self._bucket, Key=key, Body=data, ContentType=content_type
        )
        return BlobInfo(
            bucket=self._bucket, key=key, byte_size=len(data), content_type=content_type
        )

    async def get(self, key: str) -> bytes:
        try:
            response = await self._client.get_object(Bucket=self._bucket, Key=key)
        except ClientError as exc:
            if _error_code(exc) in _MISSING:
                raise BlobNotFoundError(key) from exc
            raise
        async with response["Body"] as body:
            return await body.read()

    async def stream(
        self,
        key: str,
        chunk_size: int = 1 << 20,
        start: int | None = None,
        end: int | None = None,
    ) -> AsyncIterator[bytes]:
        """Yield the object in chunks, for payloads too large to hold in memory.

        ``start``/``end`` are inclusive, as in HTTP, and issue a ranged GET so
        only that slice is transferred from the store.
        """
        request = {"Bucket": self._bucket, "Key": key}
        if start is not None or end is not None:
            request["Range"] = f"bytes={'' if start is None else start}-{'' if end is None else end}"
        try:
            response = await self._client.get_object(**request)
        except ClientError as exc:
            if _error_code(exc) in _MISSING:
                raise BlobNotFoundError(key) from exc
            raise
        # Not `async with response["Body"]`: entering it returns the underlying
        # aiohttp ClientResponse, which has no iter_chunks.
        body = response["Body"]
        try:
            async for chunk in body.iter_chunks(chunk_size):
                yield chunk
        finally:
            body.close()

    async def exists(self, key: str) -> bool:
        try:
            await self._client.head_object(Bucket=self._bucket, Key=key)
        except ClientError as exc:
            if _error_code(exc) in _MISSING:
                return False
            raise
        return True

    async def delete(self, key: str) -> bool:
        # S3 reports success for absent keys, so head first to return a truthful
        # "was it there".
        if not await self.exists(key):
            return False
        await self._client.delete_object(Bucket=self._bucket, Key=key)
        return True

    async def delete_prefix(self, prefix: str) -> int:
        deleted = 0
        paginator = self._client.get_paginator("list_objects_v2")
        batch: list[dict[str, str]] = []
        async for page in paginator.paginate(Bucket=self._bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                batch.append({"Key": obj["Key"]})
                if len(batch) == _DELETE_BATCH:
                    deleted += await self._delete_batch(batch)
                    batch = []
        deleted += await self._delete_batch(batch)
        return deleted

    async def _delete_batch(self, batch: list[dict[str, str]]) -> int:
        if not batch:
            return 0
        await self._client.delete_objects(Bucket=self._bucket, Delete={"Objects": batch})
        return len(batch)

    async def keys_under(self, prefix: str) -> list[str]:
        """Every key under a prefix."""
        paginator = self._client.get_paginator("list_objects_v2")
        found: list[str] = []
        async for page in paginator.paginate(Bucket=self._bucket, Prefix=prefix):
            found.extend(obj["Key"] for obj in page.get("Contents", []))
        return found

    async def presign_get(self, key: str, expires_in: int | None = None) -> str:
        return await self._client.generate_presigned_url(
            "get_object",
            Params={"Bucket": self._bucket, "Key": key},
            ExpiresIn=expires_in or self._presign_expiry,
        )

    async def presign_put(
        self,
        key: str,
        content_type: str | None = None,
        expires_in: int | None = None,
    ) -> str:
        params: dict[str, str] = {"Bucket": self._bucket, "Key": key}
        if content_type is not None:
            params["ContentType"] = content_type
        return await self._client.generate_presigned_url(
            "put_object", Params=params, ExpiresIn=expires_in or self._presign_expiry
        )
