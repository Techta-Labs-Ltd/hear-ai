import boto3
import httpx

from app.config import settings


class B2Storage:
    def __init__(self):
        self._client = boto3.client(
            "s3",
            endpoint_url=settings.B2_ENDPOINT_URL,
            aws_access_key_id=settings.B2_KEY_ID,
            aws_secret_access_key=settings.B2_APPLICATION_KEY,
        )

    def _public_url(self, remote_key: str) -> str:
        return f"{settings.B2_ENDPOINT_URL}/{settings.B2_BUCKET_NAME}/{remote_key}"

    def _presigned_put_url(self, remote_key: str, content_type: str = "audio/wav", expires_in: int = 900) -> str:
        return self._client.generate_presigned_url(
            "put_object",
            Params={
                "Bucket": settings.B2_BUCKET_NAME,
                "Key": remote_key,
                "ContentType": content_type,
            },
            ExpiresIn=expires_in,
        )

    def upload_file(self, local_path: str, remote_key: str, content_type: str = "audio/wav") -> str:
        signed_url = self._presigned_put_url(remote_key, content_type)
        with open(local_path, "rb") as f:
            data = f.read()
        response = httpx.put(
            signed_url,
            content=data,
            headers={"Content-Type": content_type},
            timeout=300,
        )
        response.raise_for_status()
        return self._public_url(remote_key)

    def upload_bytes(self, data: bytes, remote_key: str, content_type: str = "audio/wav") -> str:
        signed_url = self._presigned_put_url(remote_key, content_type)
        response = httpx.put(
            signed_url,
            content=data,
            headers={"Content-Type": content_type},
            timeout=300,
        )
        response.raise_for_status()
        return self._public_url(remote_key)

    def generate_url(self, key: str, expires_in: int = 3600) -> str:
        return self._client.generate_presigned_url(
            "get_object",
            Params={"Bucket": settings.B2_BUCKET_NAME, "Key": key},
            ExpiresIn=expires_in,
        )


storage = B2Storage()
