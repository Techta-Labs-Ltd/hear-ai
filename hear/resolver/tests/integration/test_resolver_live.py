import os
import random
import time

import httpx
import pytest

RESOLVER_URL = os.environ.get("RESOLVER_URL", "http://localhost:8282")

CREATORS = [
    "David Beard", "Alan Ross", "Sarah Johnson", "Mark Williams", "City Radio",
    "Tech Daily News", "London Voice", "Northern Report", "Global Sports Desk", "The Podcast Network",
    "James Miller", "Emily Clarke", "Tom Baker", "Lisa Adams", "Robert King",
    "Community Voices", "South London Radio", "Tech Today", "Health Hour", "Culture Cast",
]

CATEGORIES = [
    "sport", "news", "music", "technology", "weather", "business", "politics",
    "health", "education", "science", "culture", "food", "travel", "comedy",
    "football", "cricket", "tennis", "golf", "boxing", "running", "swimming",
    "cycling", "rugby", "basketball", "baseball", "film", "literature",
    "climate", "space", "gaming", "beauty", "fashion", "dance", "theatre",
    "horror", "fantasy", "sci-fi", "documentary", "interview", "biography",
    "mental-health", "fitness", "nutrition", "yoga", "meditation", "wellness",
]

LOCATIONS = [
    "London", "Manchester", "Birmingham", "Glasgow", "Liverpool", "Bristol",
    "Leeds", "Edinburgh", "Cardiff", "Belfast", "Newcastle", "Sheffield",
    "Brighton", "Nottingham", "Southampton", "Leicester", "Coventry", "Oxford",
    "Cambridge", "York", "Bath", "Exeter", "Norwich", "Plymouth", "Portsmouth",
]

INTENT_TEMPLATES = [
    "play {cat} from {creator}",
    "play {cat} in {loc}",
    "play {cat} from {creator} in {loc}",
    "listen to {cat} from {creator}",
    "listen to {cat} in {loc}",
    "listen to {cat} from {creator} in {loc}",
    "hear {cat} from {creator}",
    "hear {cat} in {loc}",
    "hear {cat} from {creator} near {loc}",
    "find {cat} from {creator}",
    "find {cat} in {loc}",
    "find {cat} from {creator} around {loc}",
    "show me {cat} from {creator}",
    "show me {cat} in {loc}",
    "show me {cat} from {creator} in {loc}",
    "browse {cat} from {creator}",
    "browse {cat} near {loc}",
    "start {cat} from {creator}",
    "get {cat} from {creator} in {loc}",
    "whats on for {cat} from {creator}",
    "give me {cat} in {loc}",
    "i want {cat} from {creator}",
    "i want to hear {cat} from {creator} near {loc}",
    "open {cat} by {creator}",
    "play something from {creator} about {cat}",
    "listen to something in {loc} about {cat}",
    "whats happening in {loc}",
    "tell me the latest {cat} from {creator}",
    "show me whats on near {loc} for {cat}",
    "play the latest {cat} by {creator} around {loc}",
]

MISSPELLINGS = {
    "sport": "sprt", "football": "footbal", "cricket": "crikket", "tennis": "tenis", "comedy": "comdy",
    "music": "musiq", "news": "nwes", "weather": "wether", "business": "busness", "health": "helth",
    "science": "sience", "culture": "cultur", "travel": "travle", "boxing": "boksng", "running": "runnig",
    "swimming": "swiming", "cycling": "cyclng", "rugby": "rugy", "beauty": "beuty", "fashion": "fashon",
    "dance": "dans", "theatre": "theater", "horror": "hrorr", "fantasy": "fantazy", "gaming": "gamng",
    "fitness": "fitnes", "wellness": "welness", "nutrition": "nutrtion", "meditation": "meditashn",
    "london": "lundon", "manchester": "manchestr", "birmingham": "birmingam", "glasgow": "glasgo",
    "liverpool": "livrpool", "bristol": "bristl", "leeds": "leds", "edinburgh": "edinbrugh",
    "cardiff": "cardif", "belfast": "belfst", "sheffield": "shefild", "brighton": "brightn",
    "nottingham": "notingam", "southampton": "suthamptn", "leicester": "lester", "coventry": "covntry",
    "oxford": "oxfrd", "cambridge": "cambrdge", "york": "yok", "exeter": "exetr", "norwich": "norich",
    "david": "dave", "beard": "berd", "alan": "allan", "ross": "ros", "sarah": "serah", "mark": "marc",
    "johnson": "jonsn", "williams": "wiliams", "james": "jams", "miller": "millr", "emily": "emly",
    "clarke": "clark", "thomas": "tomas", "robert": "robrt", "king": "kng", "lisa": "lees",
}


def generate_utterances(count: int = 500) -> list[dict]:
    utterances = []
    for _ in range(count):
        template = random.choice(INTENT_TEMPLATES)
        cat = random.choice(CATEGORIES)
        creator = random.choice(CREATORS)
        loc = random.choice(LOCATIONS)

        text = template.format(cat=cat, creator=creator, loc=loc)

        misspell = random.random() < 0.25
        if misspell:
            for word, bad in MISSPELLINGS.items():
                if word in text.lower():
                    text = text.lower().replace(word, bad)
                    break

        utterances.append({
            "utterance": text,
            "expected_category": cat,
            "expected_creator": creator if "{creator}" in template else None,
            "expected_location": loc if "{loc}" in template else None,
            "misspelled": misspell,
        })
    return utterances


def test_live_resolver_500_utterances():
    try:
        resp = httpx.post(
            f"{RESOLVER_URL}/resolve",
            json={"utterance": "test", "country_code": "gb"},
            timeout=10.0,
        )
        if resp.status_code >= 500:
            pytest.skip(f"Resolver not ready: HTTP {resp.status_code}")
    except Exception as e:
        pytest.skip(f"Resolver unreachable at {RESOLVER_URL}: {e}")

    utterances = generate_utterances(500)
    total = len(utterances)
    misspelled_count = sum(1 for u in utterances if u["misspelled"])
    print(f"\n{'='*80}")
    print(f"LIVE RESOLVER BENCHMARK -- {RESOLVER_URL}")
    print(f"Tests: {total} ({misspelled_count} misspelled)")
    print(f"{'='*80}")

    passed = 0
    failed = 0
    timings: list[float] = []
    misspelled_solved = 0
    errors = 0

    start_all = time.perf_counter()
    for i, test in enumerate(utterances):
        try:
            req_start = time.perf_counter()
            resp = httpx.post(
                f"{RESOLVER_URL}/resolve",
                json={"utterance": test["utterance"], "country_code": "gb"},
                timeout=10.0,
            )
            elapsed = (time.perf_counter() - req_start) * 1000
            timings.append(elapsed)

            if resp.status_code != 200:
                errors += 1
                failed += 1
                continue

            data = resp.json()
            found_cat = data.get("category", {}).get("slug") if data.get("category") else None
            found_creator = data.get("creator", {}).get("name") if data.get("creator") else None
            found_loc = data.get("location", {}).get("city") if data.get("location") else None

            ok = True
            if test["expected_category"] and found_cat != test["expected_category"]:
                ok = False
            if test["expected_creator"] and found_creator != test["expected_creator"]:
                ok = False
            if test["expected_location"] and found_loc != test["expected_location"]:
                ok = False

            if ok:
                passed += 1
                if test["misspelled"]:
                    misspelled_solved += 1
            else:
                failed += 1
        except Exception:
            errors += 1
            failed += 1

    total_elapsed = time.perf_counter() - start_all

    timings.sort()
    t = len(timings)
    if t == 0:
        pytest.fail("No successful requests")

    avg = sum(timings) / t
    p50 = timings[int(t * 0.5)]
    p90 = timings[int(t * 0.9)]
    p95 = timings[int(t * 0.95)]
    p99 = timings[int(t * 0.99)]
    p_max = timings[-1]

    print("\nRESULTS:")
    print(f"  Total:   {total}")
    print(f"  Passed:  {passed}")
    print(f"  Failed:  {failed}")
    print(f"  Errors:  {errors}")
    print(f"  Rate:    {round(passed/total*100, 1)}%")
    print(f"  Misspelled solved: {misspelled_solved}/{misspelled_count}")
    print(f"  Duration: {round(total_elapsed, 1)}s")

    print("\nLATENCY (ms):")
    print(f"  avg: {round(avg, 2)}")
    print(f"  p50: {round(p50, 2)}")
    print(f"  p90: {round(p90, 2)}")
    print(f"  p95: {round(p95, 2)}")
    print(f"  p99: {round(p99, 2)}")
    print(f"  max: {round(p_max, 2)}")
    print(f"  QPS: {round(1000/avg, 0)}")

    print(f"\n{'='*80}")
    print("SAMPLE RESULTS (first 20):")
    for i in range(min(20, total)):
        t = utterances[i]
        print(f"  {i+1:3d}. '{t['utterance'][:70]}'  misspelled={t['misspelled']}")

    print(f"\n{'='*80}")
    print("AI JUDGE (10 key misspelled cases):")
    judge = [
        ("play sprt from dave", "sport", "David Beard"),
        ("footbal in lundon", "football", "London"),
        ("crikket near glasgo", "cricket", "Glasgow"),
        ("listen to musiq from serah", "music", "Sarah Johnson"),
        ("tens in edinbrugh", "tennis", "Edinburgh"),
        ("play rugy near cardif", "rugby", "Cardiff"),
        ("find fitnes in livrpool", "fitness", "Liverpool"),
        ("hear nwes from jams millr", "news", "James Miller"),
        ("browse helth near bristl", "health", "Bristol"),
        ("play gamng from emly clark", "gaming", "Emily Clarke"),
    ]
    for utter, exp_cat, exp_ent in judge:
        start = time.perf_counter()
        resp = httpx.post(f"{RESOLVER_URL}/resolve", json={"utterance": utter, "country_code": "gb"}, timeout=10.0)
        elapsed = (time.perf_counter() - start) * 1000
        data = resp.json()
        found_cat = data.get("category", {}).get("slug") if data.get("category") else "-"
        found_ent = "-"
        etype = ""
        if data.get("creator"):
            found_ent = data["creator"].get("name", "-")
            etype = "creator"
        elif data.get("location"):
            found_ent = data["location"].get("city", "-")
            etype = "location"
        cat_ok = found_cat == exp_cat
        ent_ok = found_ent == exp_ent
        s = "PASS" if cat_ok and ent_ok else "FAIL"
        print(f"  {s} {elapsed:5.1f}ms | '{utter:45s}' -> cat={found_cat} {etype}={found_ent}")

    assert passed >= total * 0.7, f"Pass rate too low: {passed}/{total}"
    print(f"\nPASSED: {passed}/{total} ({round(passed/total*100, 1)}%)")
