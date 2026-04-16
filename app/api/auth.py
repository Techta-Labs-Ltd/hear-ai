import hmac

from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader

from app.config import settings

_api_key_scheme = APIKeyHeader(name="X-Service-Key", auto_error=False)


async def verify_service_key(api_key: str = Security(_api_key_scheme)) -> bool:
    if api_key and hmac.compare_digest(api_key, settings.AI_SERVICE_SECRET):
        return True
    raise HTTPException(status_code=401, detail="Authorization required")
