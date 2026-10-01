"""Object storage for uploaded documents.

Wraps the MinIO client so the routes never see an S3 concept. Two endpoints
matter and they are not the same one:

*   ``settings.minio_endpoint`` -- where *this process* reaches MinIO. Inside
    docker-compose that is the service name.
*   ``settings.minio_public_endpoint`` -- the host a presigned URL is signed
    for. The **browser** opens those, so signing with the compose service name
    produces links that resolve only inside the network.
"""

import hashlib
import io
from datetime import timedelta
from typing import Optional

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


def _client(endpoint: Optional[str] = None) -> Minio:
    return Minio(
        endpoint or settings.minio_endpoint,
        access_key=settings.minio_access_key,
        secret_key=settings.minio_secret_key,
        secure=settings.minio_secure,
        # Without an explicit region the client resolves it with a live
        # `?location=` call. For the public-endpoint client that host is only
        # reachable from the browser, so the lookup fails inside the container
        # and presigning 500s. See settings.minio_region.
        region=settings.minio_region,
    )


_bucket_ready = False


def ensure_bucket() -> None:
    """Create the bucket if it does not exist, once per process.

    Done here rather than by an init container: MinIO's own images were pulled
    from Docker Hub when the community edition was archived, and the
    replacement image is distroless -- no shell, no `mc`. Creating the bucket
    in-process also means a fresh volume needs no manual step.
    """
    global _bucket_ready
    if _bucket_ready:
        return
    client = _client()
    if not client.bucket_exists(settings.minio_bucket):
        client.make_bucket(settings.minio_bucket)
    _bucket_ready = True


def put_object(key: str, data: bytes, content_type: Optional[str] = None) -> None:
    ensure_bucket()
    _client().put_object(
        settings.minio_bucket,
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
        _client().remove_object(settings.minio_bucket, key)
    except S3Error as exc:
        if exc.code not in ("NoSuchKey", "NoSuchBucket"):
            raise


def presigned_url(key: str, filename: Optional[str] = None) -> str:
    """A short-lived read URL the browser can open directly.

    Signed against `minio_public_endpoint`, not the endpoint this process
    uses. Bytes never pass through FastAPI -- proxying them would make every
    preview an application request, and large files would tie up a worker.
    """
    params = {}
    if filename:
        # Makes the browser show the original filename rather than the hash.
        params["response-content-disposition"] = f'inline; filename="{filename}"'
    return _client(settings.minio_public_endpoint).presigned_get_object(
        settings.minio_bucket, key, expires=PREVIEW_URL_TTL, response_headers=params or None
    )
