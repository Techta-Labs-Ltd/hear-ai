from pydantic import BaseModel


class ResolveRequest(BaseModel):
    utterance: str
    country_code: str = "gb"


class EntityResult(BaseModel):
    id: str | None = None
    name: str | None = None
    slug: str | None = None
    city: str | None = None
    lat: str | None = None
    lng: str | None = None
    post_code: str | None = None
    confidence: float
    resolution: str | None = None


class ResolveResponse(BaseModel):
    version: int
    category: EntityResult | None = None
    creator: EntityResult | None = None
    organisation: EntityResult | None = None
    location: EntityResult | None = None
    tags: list[EntityResult] = []
    candidates: list[EntityResult] = []
    temporal: dict | None = None
    freetext: str | None = None
    action: str = "unknown"
    cache_hit: bool = False
    resolved_in_ms: float | None = None


class HealthResponse(BaseModel):
    status: str
    version: int
    index_status: str
    record_counts: dict[str, int]


class RebuildResponse(BaseModel):
    status: str
    current_version: int
    detail: str | None = None
    replicas: list[dict] | None = None


class RebuildRequest(BaseModel):
    version: int | None = None
