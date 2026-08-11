import asyncio
import hashlib
import json
import logging
import time
import uuid

from .cache.ray_cache import RayCache
from hear.config import settings
from .index.manager import DoubleBufferManager
from .models.schemas import EntityResult, HealthResponse, RebuildRequest, RebuildResponse, ResolveRequest, ResolveResponse
from .nlp.pipeline import FILLER_WORDS, NLPPipeline
from .nlp.temporal import detect_temporal
from .resolution.resolvers import resolve_category, resolve_creator, resolve_location, resolve_org, resolve_tag
from .semantic.gpu import SemanticMatcher

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

manager = DoubleBufferManager()
nlp = NLPPipeline()
cache = RayCache()
semantic = SemanticMatcher()

def _on_ready(index: dict) -> None:
    """Refresh derived state whenever a new index becomes active (startup, self-heal,
    or rebuild): NLP lookups, Tier-3 embeddings, and a Tier-2 cache flush."""
    nlp.load_aliases(index.get("creators.json", []), index.get("organisations.json", []))
    nlp.load_cities(index.get("locations.json", []))
    semantic.build(index)
    if cache.enabled:
        try:
            asyncio.get_running_loop().create_task(cache.flush_namespace())
        except RuntimeError:
            pass
    logger.info("on_ready complete version=%s", index.get("version"))


async def health() -> HealthResponse:
    active = manager.get_active()
    ready = manager.is_ready()
    counts = {
        "categories": len(active.get("categories.json", [])),
        "creators": len(active.get("creators.json", [])),
        "organisations": len(active.get("organisations.json", [])),
        "locations": len(active.get("locations.json", [])),
    }
    return HealthResponse(
        status="ok" if ready else "loading",
        version=active.get("version", 0),
        index_status="ready" if ready else "empty",
        record_counts=counts,
    )


_rebuild_running = False


async def _run_rebuild(target: int) -> None:
    global _rebuild_running
    try:
        applied = await manager.load_version(target)
        logger.info("rebuild_complete target=%d applied=%s", target, applied)
    finally:
        _rebuild_running = False


async def rebuild(req: RebuildRequest = RebuildRequest()):
    global _rebuild_running
    current = manager.get_active().get("version", 0)
    if _rebuild_running:
        return RebuildResponse(status="already_building", current_version=current)

    target = req.version
    if target is None:
        try:
            target = await manager.latest_version()
        except Exception:
            target = 0
    if not target:
        return RebuildResponse(status="no_version", current_version=current, detail="could not determine target version")

    _rebuild_running = True
    asyncio.create_task(_run_rebuild(target))
    return RebuildResponse(status="rebuild_started", current_version=current, detail=f"target v{target}")


async def apply(req: RebuildRequest = RebuildRequest()):
    ok = await manager.load_version(req.version)
    return RebuildResponse(
        status="applied" if ok else "failed",
        current_version=manager.get_active().get("version", 0),
        detail=f"v{req.version}",
    )


async def resolve_utterance(utterance: str, country: str) -> ResolveResponse:
    start = time.perf_counter()
    country = (country or "gb").lower()
    request_id = uuid.uuid4().hex[:12]
    utterance_hash = hashlib.sha256(utterance.lower().strip().encode()).hexdigest()[:16]

    cached = await cache.get_utterance(utterance, country)
    if cached is not None:
        cached["cache_hit"] = True
        cached["resolved_in_ms"] = round((time.perf_counter() - start) * 1000, 2)
        _log_request(request_id, utterance_hash, country, cached, cache_hit=True, gpu_called=False)
        return ResolveResponse(**cached)

    index = manager.get_active()
    response, gpu_called = await _resolve_core(utterance, country, index)
    response.cache_hit = False
    response.resolved_in_ms = round((time.perf_counter() - start) * 1000, 2)

    payload = response.model_dump()
    payload.pop("cache_hit", None)
    payload.pop("resolved_in_ms", None)
    await cache.set_utterance(utterance, country, payload)
    _log_request(request_id, utterance_hash, country, response.model_dump(), cache_hit=False, gpu_called=gpu_called)
    return response


def _entity_view(rec: dict, score: float, path: str | None, entity_type: str = "") -> EntityResult:
    if entity_type == "location":
        return EntityResult(
            city=rec.get("city"), lat=rec.get("lat"), lng=rec.get("lng"),
            post_code=rec.get("post_code"), confidence=round(score, 1), resolution=path,
        )
    slug = rec.get("canonical") if entity_type == "category" else rec.get("slug")
    return EntityResult(
        name=rec.get("name") or rec.get("canonical"),
        slug=slug, confidence=round(score, 1), resolution=path,
    )


async def _resolve_core(utterance: str, country: str, index: dict) -> tuple[ResolveResponse, bool]:
    parsed = nlp.process(utterance)

    cat_tokens_raw = parsed.get("category_tokens") or ""
    cat_text = " ".join(cat_tokens_raw) if isinstance(cat_tokens_raw, list) else str(cat_tokens_raw)

    creator_token = parsed.get("creator_token")
    org_token = parsed.get("org_token")
    location_token = parsed.get("location_token")

    cat_rec, cat_conf, cat_cands = None, 0.0, []
    _CAT_MIN = 95.0
    _has_pivot = bool(creator_token or org_token)
    if isinstance(cat_tokens_raw, list) and cat_tokens_raw:
        filtered = [t for t in cat_tokens_raw if t.lower() not in FILLER_WORDS]
        # When a creator/org is named (from temi), the category-side text is the topic
        # to search within that creator, not a category -- unless it is a single word
        # (boxing from temi) or the whole phrase is itself a category name. Stops a
        # trailing word like news in (resident association news from temi) being
        # mis-read as News, and skips the GPU category match (faster).
        allow_token_cat = (not _has_pivot) or len(filtered) <= 1
        if allow_token_cat:
            for tok in filtered:
                r, c, cands = resolve_category(tok, index)
                if r and c >= _CAT_MIN and (cat_rec is None or c > cat_conf):
                    cat_rec, cat_conf, cat_cands = r, c, cands
        if cat_rec is None and cat_text:
            r, c, cands = resolve_category(cat_text, index)
            if r and c >= _CAT_MIN:
                cat_rec, cat_conf, cat_cands = r, c, cands
    elif cat_text:
        cat_rec, cat_conf, cat_cands = resolve_category(cat_text, index)
        if cat_rec and cat_conf < _CAT_MIN:
            cat_rec, cat_conf, cat_cands = None, 0.0, []
    creator_rec, creator_conf, creator_cands = resolve_creator(creator_token, index) if creator_token else (None, 0.0, [])
    # Only fall back creator_token to org when creator didn't match
    org_token_or_fallback = org_token
    if not org_token_or_fallback and creator_token and creator_rec is None:
        org_token_or_fallback = creator_token
    org_rec, org_conf, org_cands = resolve_org(org_token_or_fallback, index) if org_token_or_fallback else (None, 0.0, [])
    loc_rec, loc_conf, loc_cands = resolve_location(location_token or "", index, country) if location_token else (None, 0.0, [])
    cat_path = "ram" if cat_rec else None
    creator_path = "ram" if creator_rec else None
    org_path = "ram" if org_rec else None
    loc_path = "ram" if loc_rec else None

    _min_cand_score = max(60.0, settings.RESOLVER_THRESHOLD_HIGH - 10.0)

    cat_consumed_as_location = False
    if loc_rec is None and cat_rec is None and cat_text:
        loc_rec, loc_conf, loc_cands = resolve_location(cat_text, index, country)
        if loc_rec and loc_conf >= _min_cand_score:
            loc_path = "ram"
            cat_consumed_as_location = True
        else:
            loc_rec, loc_conf, loc_cands = None, 0.0, []

    _FALLBACK_MIN = 80.0
    if cat_rec is None and cat_text:
        alt_rec, alt_conf, _ = resolve_creator(cat_text, index)
        if alt_rec and alt_conf >= _FALLBACK_MIN and creator_rec is None:
            creator_rec, creator_conf = alt_rec, alt_conf
            creator_path = "ram"
        else:
            alt_rec, alt_conf, alt_cands = resolve_org(cat_text, index)
            if alt_rec and alt_conf >= _FALLBACK_MIN and org_rec is None:
                org_rec, org_conf, org_cands = alt_rec, alt_conf, alt_cands
                org_path = "ram"

    all_candidates: list[EntityResult] = []
    if cat_rec:
        for r, s in cat_cands:
            if s >= _min_cand_score:
                all_candidates.append(_entity_view(r, s, "ram", "category"))
    if creator_rec:
        for r, s in creator_cands:
            if s >= _min_cand_score:
                all_candidates.append(_entity_view(r, s, "ram", "creator"))
    if org_rec:
        for r, s in org_cands:
            if s >= _min_cand_score:
                all_candidates.append(_entity_view(r, s, "ram", "org"))
    if loc_rec:
        for r, s in loc_cands:
            if s >= _min_cand_score:
                all_candidates.append(_entity_view(r, s, "ram", "location"))

    gpu_called = False
    if semantic.enabled:
        if cat_rec is None and not cat_consumed_as_location and cat_text and not _has_pivot:
            srec, sconf, scands = await semantic.match(cat_text, "category")
            gpu_called = True
            if srec:
                cat_rec, cat_conf, cat_path = srec, sconf, "gpu_semantic"
                for r, s in scands:
                    if s >= _min_cand_score:
                        all_candidates.append(_entity_view(r, s, "gpu_semantic", "category"))
        if creator_rec is None and creator_token:
            srec, sconf, scands = await semantic.match(creator_token, "creator")
            gpu_called = True
            if srec:
                creator_rec, creator_conf, creator_path = srec, sconf, "gpu_semantic"
                for r, s in scands:
                    if s >= _min_cand_score:
                        all_candidates.append(_entity_view(r, s, "gpu_semantic", "creator"))
        if org_rec is None and (org_token or creator_token) and not (creator_rec is not None and creator_conf >= settings.RESOLVER_THRESHOLD_HIGH):
            srec, sconf, scands = await semantic.match(org_token or creator_token, "org")
            gpu_called = True
            if srec:
                org_rec, org_conf, org_path = srec, sconf, "gpu_semantic"
                for r, s in scands:
                    if s >= _min_cand_score:
                        all_candidates.append(_entity_view(r, s, "gpu_semantic", "org"))
        if loc_rec is None and location_token:
            srec, sconf, scands = await semantic.match(location_token, "location")
            gpu_called = True
            if srec:
                loc_rec, loc_conf, loc_path = srec, sconf, "gpu_semantic"
                for r, s in scands:
                    if s >= _min_cand_score:
                        all_candidates.append(_entity_view(r, s, "gpu_semantic", "location"))

    if creator_rec and org_rec is None:
        alt_token = creator_token or (cat_text if cat_rec is None else None)
        if alt_token:
            alt_rec, alt_conf, _ = resolve_org(alt_token, index)
            if alt_rec and alt_conf >= _min_cand_score:
                all_candidates.append(_entity_view(alt_rec, alt_conf, "ram", "org"))
    if org_rec and creator_rec is None:
        alt_token = creator_token or (cat_text if cat_rec is None else None)
        if alt_token:
            alt_rec, alt_conf, _ = resolve_creator(alt_token, index)
            if alt_rec and alt_conf >= _min_cand_score:
                all_candidates.append(_entity_view(alt_rec, alt_conf, "ram", "creator"))

    tag_recs = []
    all_words = utterance.lower().strip().split()
    seen_tags: set[str] = set()
    for word in all_words:
        if len(word) < 4 or word in FILLER_WORDS:
            continue
        rec, score, _ = resolve_tag(word, index)
        if rec and score >= 85:
            slug = rec.get("slug", "")
            if slug and slug not in seen_tags:
                seen_tags.add(slug)
                tag_recs.append(EntityResult(name=rec.get("name"), slug=slug, confidence=round(score, 1), resolution="ram"))

    temporal = detect_temporal(utterance)

    freetext = None
    has_any_entity = cat_rec is not None or creator_rec is not None or org_rec is not None or loc_rec is not None
    if cat_text and not cat_consumed_as_location and (not has_any_entity or (cat_rec is None and _has_pivot)):
        freetext = cat_text

    entities = [
        (cat_rec, cat_conf), (creator_rec, creator_conf),
        (org_rec, org_conf), (loc_rec, loc_conf),
    ]
    confidences = [c for rec, c in entities if rec]
    has_entity = bool(confidences)

    if has_entity:
        action = "play" if max(confidences) >= settings.RESOLVER_THRESHOLD_HIGH else "confirm"
    elif freetext:
        action = "search"
    elif temporal:
        action = "confirm"
    else:
        action = "repeat"

    response = ResolveResponse(
        version=index.get("version", 0),
        category=_entity_view(cat_rec, cat_conf, cat_path, "category") if cat_rec else None,
        creator=_entity_view(creator_rec, creator_conf, creator_path, "creator") if creator_rec else None,
        organisation=_entity_view(org_rec, org_conf, org_path, "org") if org_rec else None,
        location=_entity_view(loc_rec, loc_conf, loc_path, "location") if loc_rec else None,
        tags=tag_recs,
        candidates=all_candidates,
        temporal=temporal,
        freetext=freetext,
        action=action,
        cache_hit=False,
        resolved_in_ms=0.0,
    )
    return response, gpu_called


async def resolve(req: ResolveRequest):
    if not manager.is_ready():
        raise RuntimeError("resolver index is not ready")
    return await resolve_utterance(req.utterance, req.country_code)


def _log_request(request_id, utterance_hash, country, resp, cache_hit, gpu_called):
    entities = {}
    for slot in ("category", "creator", "organisation", "location"):
        rec = resp.get(slot)
        if rec:
            entities[slot] = {"confidence": rec.get("confidence"), "resolution": rec.get("resolution")}
    record = {
        "request_id": request_id,
        "utterance_hash": utterance_hash,
        "country_code": country,
        "version": resp.get("version"),
        "entities": entities,
        "tag_count": len(resp.get("tags") or []),
        "action": resp.get("action"),
        "cache_hit": cache_hit,
        "gpu_called": gpu_called,
        "resolved_in_ms": resp.get("resolved_in_ms"),
    }
    logger.info("resolve %s", json.dumps(record, default=str))

