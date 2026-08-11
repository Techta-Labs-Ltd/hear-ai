from rapidfuzz import fuzz, process

THRESHOLD_HIGH = 85.0
THRESHOLD_LOW = 60.0


def fuzzy_match(token: str, records: list[dict], scorer=None, cutoff: float = THRESHOLD_LOW, limit: int = 5) -> list[tuple[dict, float]]:
    if not records or not token:
        return []
    labels = [r.get("normalized", r.get("name", "")) for r in records]
    scorer = scorer or fuzz.token_sort_ratio
    results = process.extract(token, labels, scorer=scorer, score_cutoff=cutoff, limit=limit)
    out: list[tuple[dict, float]] = []
    for label, score, idx in results:
        out.append((records[idx], score))
    return out

