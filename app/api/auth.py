import hmac
from typing import Annotated, Optional

from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer

from app.config import settings

_api_key_scheme = APIKeyHeader(name="X-Service-Key", auto_error=False)
_bearer_scheme = HTTPBearer(auto_error=False)


async def verify_service_key(
    api_key: Annotated[Optional[str], Security(_api_key_scheme)] = None,
    credentials: Annotated[Optional[HTTPAuthorizationCredentials], Security(_bearer_scheme)] = None,
) -> bool:
    if api_key and hmac.compare_digest(api_key, settings.AI_SERVICE_SECRET):
        return True
    if credentials and hmac.compare_digest(credentials.credentials, settings.AI_SERVICE_SECRET):
        return True
    raise HTTPException(status_code=401, detail="Authorization required")
