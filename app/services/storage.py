"""Object storage for uploaded documents.

Wraps an S3 client so the routes never see an S3 concept. The client is the
``minio`` package, which is a generic S3 SDK -- it speaks to Neon's storage, R2
or AWS S3 alike, and needs no MinIO server. Two endpoints matter and they are
not necessarily the same one:

*   ``settings.s3_endpoint`` -- where *this process* reaches storage.
*   ``settings.s3_public_endpoint`` -- the host a presigned URL is signed for.
    The **browser** opens those, so it must be a host the browser can resolve.
    Defaults to ``s3_endpoint``.
"""

import hashlib
import io
from datetime import timedelta
from typing import Optional, Tuple
from urllib.parse import urlsplit

from minio import Minio
from minio.error import S3Error

from app.core.config import settings

# Presigned preview links are short-lived on purpose: the URL grants read
# access to whoever holds it, so it should not outlive the page that used it.
PREVIEW_URL_TTL = timedelta(minutes=15)


def content_hash(data: bytes) -> str:
    """sha256 of the bytes -- the idempotency key and the object key both.

    Keying storage by the hash makes the upload itself idempotent: a retry
    after a failed attempt writes identical bytes to the same object rather
    than leaving a second copy behind.
    """
    return hashlib.sha256(data).hexdigest()


def object_key(digest: str) -> str:
    return f"documents/{digest}"


def _split_endpoint(endpoint: str) -> Tuple[str, bool]:
    """``https://host`` or ``host[:port]`` -> (``host[:port]``, secure).

    Providers publish the endpoint as a URL; the client wants a bare host and a
    separate TLS flag. A scheme, when present, decides TLS; otherwise
    ``s3_secure`` does.
    """
    if "://" in endpoint:
        parts = urlsplit(endpoint)
        return parts.netloc, parts.scheme == "https"
    return endpoint, settings.s3_secure


def _client(endpoint: Optional[str] = None) -> Minio:
    host, secure = _split_endpoint(endpoint or settings.s3_endpoint)
    return Minio(
        host,
        access_key=settings.s3_access_key,
        secret_key=settings.s3_secret_key,
        secure=secure,
        # Without an explicit region the client resolves it with a live
        # `?location=` call. For the public-endpoint client that host may only
        # be reachable from the browser, so the lookup can fail server-side and
        # presigning 500s. See settings.s3_region.
        region=settings.s3_region,
    )


_bucket_ready = False


def ensure_bucket() -> None:
    """Create the bucket if it does not exist, once per process.

    In production the bucket is created up front in the provider's console,
    so this is a single existence check. Creating it in-process keeps a fresh
    local setup free of a manual step.
    """
    global _bucket_ready
    if _bucket_ready:
        return
    client = _client()
    if not client.bucket_exists(settings.s3_bucket):
        client.make_bucket(settings.s3_bucket)
    _bucket_ready = True


def put_object(key: str, data: bytes, content_type: Optional[str] = None) -> None:
    ensure_bucket()
    _client().put_object(
        settings.s3_bucket,
        key,
        io.BytesIO(data),
        length=len(data),
        content_type=content_type or "application/octet-stream",
    )


def delete_object(key: str) -> None:
    """Remove an object, tolerating one that is already gone.

    Deletes run *after* the Postgres rows are committed, so a missing object
    means a previous attempt got further than it recorded -- which is the
    harmless direction. Raising here would block a delete that has already
    logically succeeded.
    """
    try:
        _client().remove_object(settings.s3_bucket, key)
    except S3Error as exc:
        if exc.code not in ("NoSuchKey", "NoSuchBucket"):
            raise


def presigned_url(key: str, filename: Optional[str] = None) -> str:
    """A short-lived read URL the browser can open directly.

    Signed against `s3_public_endpoint`, not necessarily the endpoint this
    process uses. Bytes never pass through FastAPI -- proxying them would make
    every preview an application request, and large files would tie up a
    worker.
    """
    params = {}
    if filename:
        # Makes the browser show the original filename rather than the hash.
        params["response-content-disposition"] = f'inline; filename="{filename}"'
    return _client(settings.s3_public_endpoint).presigned_get_object(
        settings.s3_bucket, key, expires=PREVIEW_URL_TTL, response_headers=params or None
    )
