from hear.config import settings
from rapidfuzz import process

from ..matching.fuzzy import THRESHOLD_LOW, fuzzy_match

LOCATION_FUZZY_CUTOFF = settings.RESOLVER_THRESHOLD_HIGH
CATEGORY_FUZZY_CUTOFF = 80.0
_CANDIDATE_LIMIT = 3
_CANDIDATE_SCORE_GAP = 10.0


def _trim_candidates(best_rec: dict | None, best_score: float, matches: list[tuple[dict, float]], gap: float = _CANDIDATE_SCORE_GAP, limit: int = _CANDIDATE_LIMIT, min_score: float = 50.0) -> list[tuple[dict, float]]:
    if not best_rec or best_score <= 0:
        return []
    floor = max(best_score - gap, min_score)
    out = []
    for rec, score in matches:
        if score < floor:
            break
        if rec is not best_rec:
            out.append((rec, score))
        if len(out) >= limit:
            break
    return out


def _fuzzy_candidates(token: str, records: list[dict], label_key: str = "normalized", cutoff: float = THRESHOLD_LOW, limit: int = 5) -> list[tuple[dict, float]]:
    if not records or not token:
        return []
    labels = []
    valid = []
    for r in records:
        lbl = r.get(label_key, r.get("name", ""))
        if lbl:
            labels.append(lbl)
            valid.append(r)
    scorer = __import__("rapidfuzz", fromlist=["fuzz"]).fuzz.token_sort_ratio
    results = process.extract(token, labels, scorer=scorer, score_cutoff=max(cutoff - 20, 30.0), limit=limit)
    return [(valid[idx], s) for _, s, idx in results]


def resolve_category(token: str, index: dict) -> tuple[dict | None, float, list[tuple[dict, float]]]:
    cats = index.get("categories.json", [])
    if not token or not cats:
        return None, 0.0, []
    normalized = token.lower().strip().replace("-", " ").replace("_", " ")
    for rec in cats:
        canon = rec.get("canonical", "").lower().replace("-", " ").replace("_", " ")
        norm = rec.get("normalized", "").lower().replace("-", " ").replace("_", " ")
        if normalized == canon or normalized == norm:
            return rec, 100.0, []
    for rec in cats:
        for phrase in rec.get("phrases", []):
            if phrase == normalized:
                return rec, 100.0, []
        for syn in rec.get("synonyms", []):
            if syn.lower().replace("-", " ").replace("_", " ") == normalized:
                return rec, 100.0, []
        stems_dict = rec.get("stems", {})
        for stem_word, target_slug in stems_dict.items():
            if normalized == stem_word and target_slug == rec.get("canonical", ""):
                return rec, 95.0, []
    if len(normalized) >= 4:
        for rec in cats:
            canonical = rec.get("canonical", "").lower()
            if canonical.startswith(normalized):
                return rec, 90.0, []
    fuzzy_matches = _fuzzy_candidates(token, cats, label_key="canonical", cutoff=CATEGORY_FUZZY_CUTOFF)
    if fuzzy_matches:
        best_rec, best_score = fuzzy_matches[0]
        candidates = _trim_candidates(best_rec, best_score, fuzzy_matches)
        return best_rec, best_score, candidates
    return None, 0.0, []


def resolve_creator(token: str, index: dict) -> tuple[dict | None, float, list[tuple[dict, float]]]:
    creators = index.get("creators.json", [])
    if not token or not creators:
        return None, 0.0, []
    normalized = token.lower().strip()
    for rec in creators:
        for alias in rec.get("aliases", []):
            if alias.lower() == normalized:
                return rec, 100.0, []
        name = rec.get("normalized", "")
        if name.startswith(normalized):
            return rec, 95.0, []
    fuzzy_matches = _fuzzy_candidates(token, creators, label_key="normalized", cutoff=THRESHOLD_LOW)
    if fuzzy_matches:
        best_rec, best_score = fuzzy_matches[0]
        candidates = _trim_candidates(best_rec, best_score, fuzzy_matches)
        return best_rec, best_score, candidates
    return None, 0.0, []


def resolve_org(token: str, index: dict) -> tuple[dict | None, float, list[tuple[dict, float]]]:
    orgs = index.get("organisations.json", [])
    if not token or not orgs:
        return None, 0.0, []
    normalized = token.lower().strip()
    for rec in orgs:
        for alias in rec.get("aliases", []):
            if alias.lower() == normalized:
                return rec, 100.0, []
        name = rec.get("normalized", "")
        if name.startswith(normalized):
            return rec, 95.0, []
    fuzzy_matches = _fuzzy_candidates(token, orgs, label_key="normalized", cutoff=THRESHOLD_LOW)
    if fuzzy_matches:
        best_rec, best_score = fuzzy_matches[0]
        candidates = _trim_candidates(best_rec, best_score, fuzzy_matches)
        return best_rec, best_score, candidates
    return None, 0.0, []


def resolve_tag(token: str, index: dict) -> tuple[dict | None, float, list[tuple[dict, float]]]:
    tags = index.get("tags.json", [])
    if not token or not tags:
        return None, 0.0, []
    normalized = token.lower().strip()
    for rec in tags:
        if rec.get("normalized", "") == normalized or rec.get("slug", "") == normalized:
            return rec, 100.0, []
        name = rec.get("normalized", "")
        if name.startswith(normalized) and len(normalized) >= 4:
            return rec, 95.0, []
    fuzzy_matches = _fuzzy_candidates(token, tags, label_key="normalized", cutoff=THRESHOLD_LOW)
    if fuzzy_matches:
        best_rec, best_score = fuzzy_matches[0]
        candidates = _trim_candidates(best_rec, best_score, fuzzy_matches)
        return best_rec, best_score, candidates
    return None, 0.0, []


def resolve_location(token: str, index: dict, country_code: str = "gb") -> tuple[dict | None, float, list[tuple[dict, float]]]:
    locs = index.get("locations.json", [])
    if not token or not locs:
        return None, 0.0, []
    country = (country_code or "").lower().strip()
    scoped = [rec for rec in locs if rec.get("country_code", "").lower() == country] if country else []
    candidates_list = scoped or locs
    normalized = token.lower().strip()
    prefix_match: dict | None = None
    for rec in candidates_list:
        city = rec.get("normalized", "")
        if city == normalized:
            return rec, 100.0, []
        if prefix_match is None and city.startswith(normalized) and len(normalized) >= 5:
            prefix_match = rec
    if prefix_match is not None:
        return prefix_match, 92.0, []
    fuzzy_matches = _fuzzy_candidates(token, candidates_list, label_key="normalized", cutoff=LOCATION_FUZZY_CUTOFF)
    if fuzzy_matches:
        best_rec, best_score = fuzzy_matches[0]
        candidates = _trim_candidates(best_rec, best_score, fuzzy_matches)
        return best_rec, best_score, candidates
    return None, 0.0, []
