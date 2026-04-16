import boto3

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

    def upload_file(self, local_path: str, remote_key: str, content_type: str = "audio/wav") -> str:
        self._client.upload_file(
            local_path,
            settings.B2_BUCKET_NAME,
            remote_key,
            ExtraArgs={"ContentType": content_type},
        )
        return self._public_url(remote_key)

    def upload_bytes(self, data: bytes, remote_key: str, content_type: str = "audio/wav") -> str:
        self._client.put_object(
            Bucket=settings.B2_BUCKET_NAME,
            Key=remote_key,
            Body=data,
            ContentType=content_type,
        )
        return self._public_url(remote_key)

    def generate_url(self, key: str, expires_in: int = 3600) -> str:
        return self._client.generate_presigned_url(
            "get_object",
            Params={"Bucket": settings.B2_BUCKET_NAME, "Key": key},
            ExpiresIn=expires_in,
        )


storage = B2Storage()
