"""MinIO (S3-compatible) wrapper for the raw document store.

Stores originals (HTML/XML/PDF) keyed by `{source}/{source_doc_id}/{content_hash}.{ext}`
so we keep every fetched version forever for legal traceability (architecture line 219).
"""

from __future__ import annotations

import io
from functools import lru_cache

from minio import Minio
from minio.error import S3Error

from regflow.common.logging import get_logger
from regflow.common.settings import get_settings

log = get_logger(__name__)


@lru_cache(maxsize=1)
def get_client() -> Minio:
    s = get_settings().minio
    return Minio(
        endpoint=s.endpoint,
        access_key=s.access_key,
        secret_key=s.secret_key.get_secret_value(),
        secure=s.secure,
    )


def ensure_bucket(bucket: str | None = None) -> str:
    bucket = bucket or get_settings().minio.raw_docs_bucket
    client = get_client()
    if not client.bucket_exists(bucket):
        client.make_bucket(bucket)
        log.info("minio.bucket_created", bucket=bucket)
    return bucket


def put_object(key: str, data: bytes, content_type: str = "application/octet-stream") -> str:
    bucket = ensure_bucket()
    client = get_client()
    client.put_object(
        bucket_name=bucket,
        object_name=key,
        data=io.BytesIO(data),
        length=len(data),
        content_type=content_type,
    )
    log.debug("minio.put", bucket=bucket, key=key, bytes=len(data))
    return f"{bucket}/{key}"


def get_object(key: str) -> bytes:
    bucket = get_settings().minio.raw_docs_bucket
    response = get_client().get_object(bucket, key)
    try:
        return response.read()
    finally:
        response.close()
        response.release_conn()


def object_exists(key: str) -> bool:
    bucket = get_settings().minio.raw_docs_bucket
    try:
        get_client().stat_object(bucket, key)
        return True
    except S3Error:
        return False
