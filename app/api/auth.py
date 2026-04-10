import hmac
from typing import Optional, Annotated

from fastapi import Depends, Header, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from app.config import settings

bearer = HTTPBearer(auto_error=False)


async def verify_service_key(
    x_service_key: Annotated[Optional[str], Header()] = None,
    credentials: Annotated[Optional[HTTPAuthorizationCredentials], Depends(bearer)] = None,
) -> bool:
    if x_service_key:
        if hmac.compare_digest(x_service_key, settings.AI_SERVICE_SECRET):
            return True
        raise HTTPException(status_code=401, detail="Invalid service key")

    if credentials:
        if hmac.compare_digest(credentials.credentials, settings.AI_SERVICE_SECRET):
            return True
        raise HTTPException(status_code=401, detail="Invalid token")

    raise HTTPException(status_code=401, detail="Authorization required")
