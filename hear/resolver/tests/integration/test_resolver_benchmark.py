import json
import random
import time

import pytest

from hear.resolver.resolution.resolvers import resolve_category, resolve_creator, resolve_location

SAMPLE_DATA = {
    "CATEGORY_NAMES": [
        "news", "sports", "music", "technology", "weather", "business", "politics",
        "health", "education", "science", "culture", "food", "travel", "faith",
        "football", "cricket", "tennis", "golf", "boxing", "running", "swimming",
        "cycling", "rugby", "basketball", "baseball", "comedy", "film", "literature",
        "climate", "space", "robotics", "gaming", "esports", "beauty", "fashion",
        "dance", "theatre", "anime", "horror", "fantasy", "sci-fi", "poetry",
        "documentary", "interview", "debate", "biography", "true-crime", "mental-health",
        "fitness", "nutrition", "yoga", "meditation", "wellness", "addiction", "recovery",
    ],
    "CATEGORY_NAMES_MAP": {n: n.replace("-", " ").title() for n in [
        "news", "sports", "music", "technology", "weather", "business", "politics",
        "health", "education", "science", "culture", "food", "travel", "faith",
        "football", "cricket", "tennis", "golf", "boxing", "running", "swimming",
        "cycling", "rugby", "basketball", "baseball", "comedy", "film", "literature",
        "climate", "space", "robotics", "gaming", "esports", "beauty", "fashion",
        "dance", "theatre", "anime", "horror", "fantasy", "sci-fi", "poetry",
        "documentary", "interview", "debate", "biography", "true-crime", "mental-health",
        "fitness", "nutrition", "yoga", "meditation", "wellness", "addiction", "recovery",
    ]},
    "CREATORS": [
        {"name": "David Beard", "aliases": ["david", "beard", "DB"]},
        {"name": "Alan Ross", "aliases": ["alan", "ross", "AR"]},
        {"name": "City Radio", "aliases": ["city", "radio"]},
        {"name": "Tech Daily News", "aliases": ["tech", "tdn"]},
        {"name": "Sarah Johnson", "aliases": ["sarah", "johnson"]},
        {"name": "Mark Williams", "aliases": ["mark", "williams", "mw"]},
        {"name": "London Voice", "aliases": ["london", "lv"]},
        {"name": "Northern Report", "aliases": ["north", "nr"]},
        {"name": "Global Sports Desk", "aliases": ["global", "gsd", "sports"]},
        {"name": "The Podcast Network", "aliases": ["podcast", "tpn", "network"]},
    ],
    "ORGANIZATIONS": [
        {"name": "TNF", "aliases": ["the news foundation", "tnf"]},
        {"name": "TechDaily", "aliases": ["tech daily", "td"]},
        {"name": "BBC", "aliases": ["bbc", "british broadcasting"]},
        {"name": "Hear Media", "aliases": ["hear", "hm"]},
        {"name": "City Press", "aliases": ["city", "cp", "press"]},
    ],
    "LOCATIONS": [
        {"city": "London", "country_code": "GB", "address": "Greater London"},
        {"city": "Manchester", "country_code": "GB", "address": "Greater Manchester"},
        {"city": "Birmingham", "country_code": "GB", "address": "West Midlands"},
        {"city": "Glasgow", "country_code": "GB", "address": "Scotland"},
        {"city": "Liverpool", "country_code": "GB", "address": "Merseyside"},
        {"city": "Bristol", "country_code": "GB", "address": "South West"},
        {"city": "Leeds", "country_code": "GB", "address": "West Yorkshire"},
        {"city": "Edinburgh", "country_code": "GB", "address": "Scotland"},
        {"city": "Cardiff", "country_code": "GB", "address": "Wales"},
        {"city": "Belfast", "country_code": "GB", "address": "Northern Ireland"},
        {"city": "Newcastle", "country_code": "GB", "address": "Tyne and Wear"},
        {"city": "Sheffield", "country_code": "GB", "address": "South Yorkshire"},
        {"city": "Brighton", "country_code": "GB", "address": "East Sussex"},
        {"city": "Nottingham", "country_code": "GB", "address": "East Midlands"},
        {"city": "Southampton", "country_code": "GB", "address": "Hampshire"},
        {"city": "Lagos", "country_code": "NG", "address": "Lagos State"},
        {"city": "Abuja", "country_code": "NG", "address": "FCT"},
        {"city": "Port Harcourt", "country_code": "NG", "address": "Rivers State"},
    ],
}

FILLER = frozenset({"play", "listen", "hear", "watch", "find", "show", "give", "get", "open", "browse", "whats",
                     "start", "tell", "me", "please", "the", "a", "an", "to", "for", "some",
                     "can", "you", "i", "want", "like", "need", "looking", "something",
                     "about", "with", "my", "now", "just", "let", "id"})


def parse_utterance(text: str) -> dict:
    tokens = [t for t in text.lower().strip().split() if t not in FILLER]
    seen_creator_prep = False
    seen_loc_prep = False
    cat_parts: list[str] = []
    creator_parts: list[str] = []
    loc_parts: list[str] = []
    buf: list[str] = []

    for word in tokens:
        if word in ("from", "by"):
            cat_parts.extend(buf)
            buf = []
            seen_creator_prep = True
        elif word in ("in", "near", "around", "at"):
            if seen_creator_prep:
                creator_parts = buf
            else:
                cat_parts.extend(buf)
            buf = []
            seen_loc_prep = True
        else:
            buf.append(word)

    if buf:
        if seen_loc_prep:
            loc_parts = buf
        elif seen_creator_prep:
            creator_parts = buf
        else:
            cat_parts.extend(buf)

    return {"category_text": " ".join(cat_parts) if cat_parts else "", "creator_text": " ".join(creator_parts) if creator_parts else None, "location_text": " ".join(loc_parts) if loc_parts else None}


MISSPELLINGS = {
    "sport": "sprt", "football": "footbal", "cricket": "crikket", "tennis": "tenis", "comedy": "comdy",
    "music": "musik", "news": "nwes", "weather": "wether", "business": "busness", "health": "helth",
    "boxing": "boksng", "running": "runnig", "beauty": "beuty", "horror": "hrorr",
    "fantasy": "fantazy", "fitness": "fitnes", "wellness": "welness", "recovery": "recoery",
    "london": "lundon", "manchester": "manchestr", "birmingham": "birmingam", "glasgow": "glasgo",
    "liverpool": "livrpool", "bristol": "bristl", "edinburgh": "edinbrugh", "cardiff": "cardif", "belfast": "belfst",
    "david": "dave", "beard": "berd", "alan": "allan", "ross": "ros", "sarah": "serah", "mark": "marc", "johnson": "jonsn", "williams": "wiliams",
}


def generate_misspell(text: str) -> list[str]:
    result = []
    for word, bad in MISSPELLINGS.items():
        if word in text.lower():
            result.append(text.lower().replace(word, bad))
    return result


def test_resolver_benchmark():
    index = {
        "version": 1,
        "categories.json": [
            {
                "canonical": slug,
                "normalized": slug.replace("-", " "),
                "phrases": [slug.replace("-", " ")],
                "synonyms": [],
                "stems": {},
            }
            for slug in SAMPLE_DATA["CATEGORY_NAMES"]
        ],
        "creators.json": [
            {
                **record,
                "normalized": record["name"].lower(),
                "aliases": [alias.lower() for alias in record["aliases"]],
            }
            for record in SAMPLE_DATA["CREATORS"]
        ],
        "organisations.json": [
            {
                **record,
                "normalized": record["name"].lower(),
                "aliases": [alias.lower() for alias in record["aliases"]],
            }
            for record in SAMPLE_DATA["ORGANIZATIONS"]
        ],
        "locations.json": [
            {**record, "normalized": record["city"].lower()}
            for record in SAMPLE_DATA["LOCATIONS"]
        ],
    }

    tests = [
        ("play sport from david", "sports", "David Beard", None),
        ("listen to news from alan ross", "news", "Alan Ross", None),
        ("play football in london", "football", None, "London"),
        ("hear music from sarah in manchester", "music", "Sarah Johnson", "Manchester"),
        ("start podcast from the podcast network", None, "The Podcast Network", None),
        ("play tech from tech daily news in birmingham", "technology", "Tech Daily News", "Birmingham"),
        ("browse comedy", "comedy", None, None),
        ("find running events near glasgow", "running", None, "Glasgow"),
        ("play cricket from global sports desk", "cricket", "Global Sports Desk", None),
        ("show me boxing in liverpool", "boxing", None, "Liverpool"),
    ]

    all_cases = []
    for utter, cat, creator, loc in tests:
        all_cases.append((utter, cat, creator, loc, False))
        for m in generate_misspell(utter):
            all_cases.append((m, cat, creator, loc, True))

    print(f"\n{'='*90}")
    print(f"RESOLVER BENCHMARK -- {len(all_cases)} tests ({len([a for a in all_cases if a[4]])} misspelled)")
    print(f"{'='*90}")

    passed = 0
    failed = 0
    timings: list[float] = []

    for utter, exp_cat, exp_creator, exp_loc, misspelled in all_cases:
        parsed = parse_utterance(utter)
        start = time.perf_counter()
        cat_rec, _, _ = resolve_category(parsed["category_text"], index) if parsed["category_text"] else (None, 0.0, [])
        creator_rec, _, _ = resolve_creator(parsed["creator_text"], index) if parsed["creator_text"] else (None, 0.0, [])
        loc_rec, _, _ = resolve_location(parsed["location_text"], index) if parsed["location_text"] else (None, 0.0, [])
        elapsed = (time.perf_counter() - start) * 1000
        timings.append(elapsed)

        found_cat = cat_rec.get("canonical") if cat_rec else None
        found_creator = creator_rec.get("name") if creator_rec else None
        found_loc = loc_rec.get("city") if loc_rec else None

        ok = True
        if exp_cat and found_cat != exp_cat:
            ok = False
        if exp_creator and found_creator != exp_creator:
            ok = False
        if exp_loc and found_loc != exp_loc:
            ok = False
        if ok:
            passed += 1
        else:
            failed += 1

    timings.sort()
    t = len(timings)
    avg = sum(timings) / t
    p50 = timings[int(t * 0.5)]
    p90 = timings[int(t * 0.9)]
    p99 = timings[int(t * 0.99)]
    p_max = timings[-1]

    print(f"Passed: {passed} | Failed: {failed} | Rate: {round(passed/t*100, 1)}%")
    print(f"Latency (ms) -- avg:{round(avg,2)} p50:{round(p50,2)} p90:{round(p90,2)} p99:{round(p99,2)} max:{round(p_max,2)} QPS:{round(1000/avg, 0)}")

    print(f"\n{'='*90}")
    print("AI JUDGE -- Sample Verification")
    print(f"{'='*90}")
    judge = [
        ("play sport from david", "sports", "David Beard"),
        ("sprt from dave", "sports", "David Beard"),
        ("footbal in lundon", "football", "London"),
        ("crikket near glasgo", "cricket", "Glasgow"),
        ("listen to musiq from serah", "music", "Sarah Johnson"),
        ("play comedy from the podcast network", "comedy", "The Podcast Network"),
        ("news from alan ross", "news", "Alan Ross"),
        ("tens in edinbrugh", "tennis", "Edinburgh"),
    ]
    for utter, exp_cat, exp_ent in judge:
        parsed = parse_utterance(utter)
        start = time.perf_counter()
        cat_rec, _, _ = resolve_category(parsed["category_text"], index) if parsed["category_text"] else (None, 0.0, [])
        crec, _, _ = resolve_creator(parsed["creator_text"], index) if parsed["creator_text"] else (None, 0.0, [])
        lrec, _, _ = resolve_location(parsed["location_text"], index) if parsed["location_text"] else (None, 0.0, [])
        elapsed = (time.perf_counter() - start) * 1000
        found_cat = cat_rec.get("canonical") if cat_rec else "-"
        found_ent = "-"
        if crec:
            found_ent = crec.get("name", "-")
        elif lrec:
            found_ent = lrec.get("city", "-")
        cat_ok = found_cat == exp_cat
        ent_ok = found_ent == exp_ent
        s = "PASS" if cat_ok and ent_ok else "FAIL"
        print(f"  {s} {elapsed:5.1f}ms | '{utter:40s}' -> cat={found_cat} ent={found_ent}")

    print(f"\n{'='*90}")
    print("LOAD TEST -- 500 sequential resolves")
    print(f"{'='*90}")
    lt: list[float] = []
    for _ in range(500):
        utter, _, _, _ = random.choice(tests)
        parsed = parse_utterance(utter)
        start = time.perf_counter()
        resolve_category(parsed["category_text"], index) if parsed["category_text"] else (None, 0.0, [])
        resolve_creator(parsed["creator_text"], index) if parsed["creator_text"] else (None, 0.0, [])
        resolve_location(parsed["location_text"], index) if parsed["location_text"] else (None, 0.0, [])
        lt.append((time.perf_counter() - start) * 1000)
    lt.sort()
    n = len(lt)
    print(f"Samples: {n}")
    print(f"avg:{round(sum(lt)/n,2)}ms p50:{round(lt[int(n*.5)],2)}ms p90:{round(lt[int(n*.9)],2)}ms p99:{round(lt[int(n*.99)],2)}ms max:{round(lt[-1],2)}ms QPS:{round(1000/(sum(lt)/n),0)}")

    assert passed >= len(all_cases) * 0.5, f"Pass rate too low: {passed}/{len(all_cases)}"
