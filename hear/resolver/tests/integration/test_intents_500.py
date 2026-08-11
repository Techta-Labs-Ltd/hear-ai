import os
import random
import time

import httpx

RESOLVER_URL = os.environ.get("RESOLVER_URL", "http://localhost:8282")

CATEGORIES = [
    "news", "sports", "entertainment", "music", "comedy", "weather",
    "business", "politics", "technology", "health", "education", "science",
    "culture", "food", "travel", "faith",
]

CREATORS = [
    "David Beard", "Alan Ross", "Sarah Johnson", "Mark Williams", "City Radio",
    "Tech Daily News", "London Voice", "Northern Report", "Global Sports Desk",
    "The Podcast Network", "James Miller", "Emily Clarke", "Tom Baker",
    "Lisa Adams", "Robert King", "Community Voices", "South London Radio",
    "Tech Today", "Health Hour", "Culture Cast",
]

LOCATIONS = [
    "London", "Manchester", "Birmingham", "Glasgow", "Liverpool", "Bristol",
    "Leeds", "Edinburgh", "Cardiff", "Belfast", "Newcastle", "Sheffield",
    "Brighton", "Nottingham", "Southampton", "Leicester", "Coventry", "Oxford",
    "Cambridge", "York",
]

SKIP_FEEDBACK = ["skip", "never mind", "no thanks", "ignore that", "skip feedback",
    "move on", "don't bother", "I don't want to rate", "no comment", "pass",
    "skip the rating", "carry on", "I'd rather not say", "just play the next one",
    "whatever", "doesn't matter", "skip it", "not bothered", "can't be bothered",]

NOW_PLAYING_PHRASES = ["who is the creator", "who made this", "who created this",
    "who is the author", "who recorded this", "who is speaking", "credit",
    "who is this by", "who is the narrator", "tell me about the creator",
    "who published this", "who am I listening to", "who is talking",
    "who wrote this", "whose voice is this", "who's this",
    "what's this about", "what is this about", "what is this content about",
    "tell me about this recording", "what am I listening to",
    "what's it about", "describe this recording",
    "enjoyed", "I enjoyed it", "I liked it", "that was great", "loved it",
    "great", "good one", "brilliant", "amazing", "fantastic",
    "that was good", "really good", "excellent",
    "it was okay", "not bad", "alright", "so so", "average",
    "did not enjoy it", "bad", "not for me", "terrible", "awful",
    "didn't enjoy it", "rubbish", "boring",
    "follow this creator", "follow them", "subscribe to this creator",
    "follow", "subscribe", "follow this person",
    "unfollow this creator", "unfollow them", "unsubscribe",
    "report this content", "flag this content", "report",
    "report this creator", "flag this creator", "report the creator",
    "what's trending", "what is trending", "what's popular",
    "what's on", "what's available", "what's new", "recommend something",
    "what's been published", "what are people listening to",
]

PLAYBACK_CTRL = [
    "pause", "resume", "stop", "stop playing", "cancel", "start over",
    "restart", "from the beginning", "repeat", "play again", "say that again",
    "replay", "one more time", "again", "speed up", "slow down",
    "go faster", "go slower", "faster", "slower",
    "rewind", "fast forward", "skip back", "skip forward", "skip ahead",
    "jump back", "jump forward",
]

TEMPORAL_MODIFIERS = ["", " latest", " newest", " recent", " today", " this week"]


def generate_utterances(count: int = 500) -> list[dict]:
    utterances = []
    templates = []

    # PlayContentIntent — free-form topic
    for cat in CATEGORIES:
        for tmpl in [
            "play {topic}",
            "play some {topic}",
            "I want {topic}",
            "give me {topic}",
            "something about {topic}",
            "play me something on {topic}",
            "do you have {topic}",
            "find something about {topic}",
        ]:
            templates.append((tmpl.replace("{topic}", cat), cat, "topic"))

    # PlayByCategoryIntent — explicit category
    for cat in CATEGORIES:
        for tmpl in [
            "play{mod} {cat}",
            "play some{mod} {cat}",
            "play the{mod} {cat}",
        ]:
            for m in TEMPORAL_MODIFIERS:
                text = tmpl.replace("{mod}", m).replace("{cat}", cat).strip()
                templates.append((text, cat, None))

    # PlayByCreatorIntent
    for creator in CREATORS:
        for tmpl in [
            "play from {cre}",
            "play something from {cre}",
            "play me something from {cre}",
            "play {cre}",
            "something from {cre}",
            "find {cre}",
            "play the latest from {cre}",
            "hear the latest from {cre}",
        ]:
            templates.append((tmpl.replace("{cre}", creator), creator, "creator"))

    # PlayByOrganizationIntent
    for org in ["bbc", "cnn", "the guardian", "daily mail", "the times",
                 "global sports desk", "the podcast network", "tech today",
                 "community voices", "south london radio"]:
        for tmpl in [
            "play from {org}",
            "play content from {org}",
            "play from the {org}",
            "something from {org}",
            "show me content from {org}",
            "play me something from {org}",
        ]:
            templates.append((tmpl.replace("{org}", org), org, "org"))

    # Location-based
    for loc in LOCATIONS:
        for tmpl_pre in [" in {loc}", " near {loc}", " around {loc}"]:
            for cat in random.sample(CATEGORIES, min(3, len(CATEGORIES))):
                for m in TEMPORAL_MODIFIERS:
                    text = f"play{m} {cat}{tmpl_pre.replace('{loc}', loc)}"
                    templates.append((text.strip(), cat, "loc"))
                    break

    # Compound: category + creator + location
    for _ in range(30):
        cat = random.choice(CATEGORIES)
        cre = random.choice(CREATORS)
        loc = random.choice(LOCATIONS)
        templates.append((f"play {cat} from {cre} in {loc}", cat, "compound"))
        templates.append((f"play {cat} from {cre}", cat, "compound"))
        templates.append((f"play {cat} in {loc}", cat, "compound"))

    # System intents
    for phrase in SKIP_FEEDBACK:
        templates.append((phrase, None, "system"))
    for phrase in NOW_PLAYING_PHRASES:
        templates.append((phrase, None, "system"))
    for phrase in PLAYBACK_CTRL:
        templates.append((phrase, None, "system"))

    # Help / navigation
    for phrase in [
        "help", "what can I say", "what can you do", "home", "go home",
        "main menu", "show me more", "what are the next ones", "more recordings",
        "what else did you find", "browse", "what's on",
    ]:
        templates.append((phrase, None, "system"))

    # Temporal patterns
    for cat in CATEGORIES[:6]:
        for t in ["today", "yesterday", "this week", "last week",
                   "this month", "last month", "3 days ago",
                   "this morning", "last night", "this weekend"]:
            templates.append((f"play {cat} from {t}", cat, "temporal"))
            templates.append((f"play {t} {cat}", cat, "temporal"))

    # Free-form / search queries
    for phrase in [
        "bat and ball game", "space travel", "cooking show",
        "learn a language", "morning meditation", "bedtime story",
        "world news today", "football highlights", "music from the 90s",
        "gardening tips", "tech reviews", "book club",
        "true crime documentary", "sports news", "financial advice",
        "stock market update", "weather forecast", "traffic update",
        "local news", "celebrity gossip", "movie reviews",
    ]:
        templates.append((phrase, None, "freetext"))

    # Misspellings
    misspellings = {
        "sport": "sprt", "football": "footbal", "cricket": "crikket",
        "tennis": "tenis", "comedy": "comdy", "music": "musiq",
        "news": "nwes", "weather": "wether", "business": "busness",
        "health": "helth", "science": "sience", "culture": "cultur",
        "travel": "travle", "london": "lundon", "manchester": "manchestr",
        "birmingham": "birminghm", "glasgow": "glasgo", "liverpool": "livrpool",
        "bristol": "bristl", "edinburgh": "edinbrugh", "cardiff": "cardif",
        "brighton": "brightn", "nottingham": "notingam",
        "david": "dave", "sarah": "serah", "james": "jams",
    }
    for cat in CATEGORIES[:8]:
        t = random.choice(CATEGORIES)
        for cre in CREATORS[:5]:
            misspelled_cat = misspellings.get(cat.lower(), "" if random.random() > 0.5 else cat.lower())
            if misspelled_cat:
                templates.append((f"play {misspelled_cat}", cat, "topic"))
                templates.append((f"play {misspelled_cat} from {cre.lower()}", cat, "topic"))
                break
        for loc in LOCATIONS[:5]:
            misspelled_loc = misspellings.get(loc.lower(), "")
            if misspelled_loc:
                templates.append((f"play news in {misspelled_loc}", "news", "loc"))
                break

    # Shuffle and pick
    random.shuffle(templates)
    selected = templates[:count]

    for tmpl_str, expected, etype in selected:
        utterances.append({
            "text": tmpl_str,
            "expected_category": expected if etype in ("topic", "compound", "loc") else None,
            "expected_creator": expected if etype == "creator" else None,
            "expected_location": expected if etype == "loc" else None,
            "intent_type": etype or "unknown",
        })

    while len(utterances) < count:
        cat = random.choice(CATEGORIES)
        utterances.append({
            "text": f"play {cat}",
            "expected_category": cat,
            "expected_creator": None,
            "expected_location": None,
            "intent_type": "topic",
        })

    random.shuffle(utterances)
    return utterances[:count]


def test_500_intents():
    try:
        resp = httpx.post(
            f"{RESOLVER_URL}/resolve",
            json={"utterance": "test", "country_code": "gb"},
            timeout=10.0,
        )
        if resp.status_code >= 500:
            import pytest; pytest.skip(f"Resolver not ready: HTTP {resp.status_code}")
    except Exception as e:
        import pytest; pytest.skip(f"Resolver unreachable at {RESOLVER_URL}: {e}")

    utterances = generate_utterances(500)
    total = len(utterances)
    print(f"\n{'='*90}")
    print(f"INTENT-BASED RESOLVER TEST -- {RESOLVER_URL}")
    print(f"Tests: {total}")
    print(f"{'='*90}")

    passed = 0
    failed = 0
    timings: list[float] = []
    action_counts: dict[str, int] = {}
    freetext_count = 0
    candidate_counts: list[int] = []
    failures: list[dict] = []

    for i, test in enumerate(utterances):
        try:
            req_start = time.perf_counter()
            resp = httpx.post(
                f"{RESOLVER_URL}/resolve",
                json={"utterance": test["text"], "country_code": "gb"},
                timeout=10.0,
            )
            elapsed = (time.perf_counter() - req_start) * 1000
            timings.append(elapsed)

            if resp.status_code != 200:
                failed += 1
                continue

            data = resp.json()
            action_counts[data.get("action", "unknown")] = action_counts.get(data.get("action", "unknown"), 0) + 1
            if data.get("freetext"):
                freetext_count += 1
            candidate_counts.append(len(data.get("candidates", [])))

            found_cat = data.get("category", {}).get("slug") if data.get("category") else None
            found_creator = data.get("creator", {}).get("name") if data.get("creator") else None
            found_loc = data.get("location", {}).get("city") if data.get("location") else None
            found_tags = [t.get("slug") for t in data.get("tags", [])]

            ok = True
            if test["expected_category"] and found_cat != test["expected_category"]:
                ok = False
            if test["expected_creator"] and found_creator != test["expected_creator"]:
                ok = False
            if test["expected_location"] and found_loc != test["expected_location"]:
                ok = False
            if ok:
                passed += 1
            else:
                failed += 1
                expected = test.get("expected_category") or test.get("expected_creator") or test.get("expected_location") or "-"
                failures.append({
                    "text": test["text"], "exp": expected,
                    "cat": found_cat, "cre": found_creator, "loc": found_loc,
                    "tags": found_tags, "action": data.get("action"),
                })
        except Exception:
            failed += 1

    timings.sort()
    t = len(timings)
    if t == 0:
        import pytest; pytest.fail("No successful requests")

    avg = sum(timings) / t
    p50 = timings[int(t * 0.5)]
    p90 = timings[int(t * 0.9)]
    p95 = timings[int(t * 0.95)]
    p99 = timings[int(t * 0.99)]
    p_max = timings[-1]

    print(f"\nRESULTS:")
    print(f"  Total:   {total}")
    print(f"  Passed:  {passed}")
    print(f"  Failed:  {failed}")
    print(f"  Rate:    {round(passed/total*100, 1)}%")
    print(f"\nACTIONS:")
    for action, count in sorted(action_counts.items(), key=lambda x: -x[1]):
        print(f"  {action}: {count}")
    print(f"\nFREETEXT:")
    print(f"  utterances with freetext: {freetext_count}/{total}")
    print(f"\nCANDIDATES:")
    cand_any = sum(1 for c in candidate_counts if c > 0)
    if candidate_counts:
        print(f"  utterances with candidates: {cand_any}/{total}")
        print(f"  avg candidates: {sum(candidate_counts)/len(candidate_counts):.1f}")
        print(f"  max candidates: {max(candidate_counts)}")

    print(f"\nLATENCY (ms):")
    print(f"  avg: {round(avg, 2)}")
    print(f"  p50: {round(p50, 2)}")
    print(f"  p90: {round(p90, 2)}")
    print(f"  p95: {round(p95, 2)}")
    print(f"  p99: {round(p99, 2)}")
    print(f"  max: {round(p_max, 2)}")
    print(f"  QPS: {round(1000/avg, 0)}")

    print(f"\nINTENT BREAKDOWN:")
    intent_counts: dict[str, int] = {}
    for u in utterances:
        intent_counts[u["intent_type"]] = intent_counts.get(u["intent_type"], 0) + 1
    for itype, count in sorted(intent_counts.items(), key=lambda x: -x[1]):
        print(f"  {itype}: {count}")

    print(f"\n{'='*90}")
    print("SAMPLE RESULTS (first 25):")
    for i in range(min(25, total)):
        u = utterances[i]
        print(f"  {i+1:3d}. [{u['intent_type']:10s}] '{u['text'][:70]}'  exp={u.get('expected_category') or u.get('expected_creator') or '-'}")

    print(f"\n{'='*90}")
    print(f"FAILURES ({len(failures)}):")
    if failures:
        print(f"  {'utt':30s} {'cat':12s} {'cre':20s} {'loc':12s} {'tags':15s} {'action':10s} exp")
        print(f"  {'-'*30} {'-'*12} {'-'*20} {'-'*12} {'-'*15} {'-'*10} {'-'*10}")
        for f in failures:
            tags_str = ','.join(f["tags"]) if f["tags"] else "∅"
            print(f"  {f['text'][:30]:30s} {str(f['cat'] or '∅'):12s} {str(f['cre'] or '∅'):20s} {str(f['loc'] or '∅'):12s} {tags_str:15s} {f['action']:10s} {f['exp']}")

    print(f"\n{'='*90}")
    print("ACTION SAMPLES (first of each action):")
    seen_actions: set[str] = set()
    for u in utterances:
        if len(seen_actions) >= 5:
            break
        resp = httpx.post(f"{RESOLVER_URL}/resolve", json={"utterance": u["text"], "country_code": "gb"}, timeout=10.0)
        d = resp.json()
        act = d.get("action", "?")
        if act in seen_actions:
            continue
        seen_actions.add(act)
        freetext = d.get("freetext")
        cat = d.get("category", {}).get("slug") if d.get("category") else "∅"
        cre = d.get("creator", {}).get("name") if d.get("creator") else "∅"
        loc = d.get("location", {}).get("city") if d.get("location") else "∅"
        tags = [t.get("slug") for t in d.get("tags", [])]
        cands = [(c.get("name") or c.get("city"), round(c["confidence"], 1)) for c in d.get("candidates", [])]
        print(f"  action={act:10s} cat={cat:12s} cre={cre:20s} loc={loc:12s} tags={','.join(tags) if tags else '∅':12s} ft={str(freetext):15s} cand={cands}  | '{u['text']}'")

    print(f"\n{'='*90}")
    print("ALL TAGS SEEN:")
    all_tags: set[str] = set()
    for u in utterances:
        resp = httpx.post(f"{RESOLVER_URL}/resolve", json={"utterance": u["text"], "country_code": "gb"}, timeout=10.0)
        d = resp.json()
        for t in d.get("tags", []):
            all_tags.add(t.get("slug", ""))
    if all_tags:
        print(f"  {', '.join(sorted(all_tags))}")
    else:
        print("  (none)")

    import pytest
    assert passed >= total * 0.5, f"Pass rate too low: {passed}/{total}"
    print(f"\nPASSED: {passed}/{total} ({round(passed/total*100, 1)}%)")
