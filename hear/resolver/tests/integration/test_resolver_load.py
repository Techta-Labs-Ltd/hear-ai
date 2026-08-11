import asyncio
import random
import time

import httpx

URL = "https://api.hear.surf/api/v1/alexa/resolve"
CONCURRENT = 500

TEMPLATES = [
    "play sport from david in london",
    "listen to news from alan ross in manchester",
    "hear music from sarah in birmingham",
    "play football near glasgow",
    "find cricket from global sports desk",
    "show me comedy",
    "browse technology from tech daily news",
    "play running in liverpool",
    "hear news from mark williams in bristol",
    "listen to tennis near edinburgh",
    "play golf from northern report",
    "find boxing in cardiff",
    "start podcast from the podcast network",
    "show me rugby near belfast",
    "play gaming from emily clarke",
    "hear fitness in brighton",
    "browse beauty from lisa adams",
    "listen to horror near nottingham",
    "find sci-fi from robert king in southampton",
    "show me interview in leeds",
    "play documentary from community voices",
    "hear literature from tom baker near newcastle",
    "browse health in sheffield",
    "find fashion from london voice",
    "play theatre near oxford",
    "listen to biography from culture cast in cambridge",
    "hear cycling near coventry",
    "show me swimming from city radio in york",
    "browse nutrition near exeter",
    "play meditation from health hour in norwich",
    "find yoga near bath",
    "listen to wellness from south london radio",
    "hear climate in portsmouth",
    "show me space from tech today near plymouth",
    "browse robotics from james miller",
    "play film in leicester",
    "listen to education from alan ross near cardiff",
    "hear politics in belfast",
    "show me business from city press",
    "browse weather in london",
    "play baseball from global sports desk near manchester",
    "listen to dance from sarah near birmingham",
    "hear animation from emily clarke in glasgow",
    "show me history from robert king near liverpool",
    "browse cooking from lisa adams in bristol",
    "play travel from tom baker near leeds",
    "listen to photography from mark williams in edinburgh",
    "hear religion from james miller near newcastle",
    "show me law from city radio in cardiff",
    "browse economics from northern report near belfast",
]

ALEXA_STYLE = [
    "alexa play sport from david",
    "alexa listen to news from alan ross in london",
    "alexa hear music near manchester",
    "alexa play football from global sports desk in glasgow",
    "alexa find cricket matches in london",
    "alexa show me comedy from the podcast network",
    "alexa browse technology updates from tech daily news",
    "alexa start running playlist in liverpool",
    "alexa get news from mark williams near bristol",
    "alexa open tennis highlights from sports desk",
    "alexa tell me the latest golf from northern report in edinburgh",
    "alexa give me boxing news from city radio",
    "alexa what's on for rugby near belfast",
    "alexa play the latest gaming from emily clarke",
    "alexa i want to hear fitness tips in brighton",
    "alexa listen to horror stories near nottingham",
    "alexa show me sci-fi movies from robert king in southampton",
    "alexa browse interview podcasts in leeds",
    "alexa play documentary from community voices near newcastle",
    "alexa hear literature from tom baker",
    "alexa show me health news in sheffield",
    "alexa browse fashion from london voice near oxford",
    "alexa play theatre shows from culture cast in cambridge",
    "alexa listen to biography podcasts",
    "alexa hear cycling updates near coventry",
    "alexa find swimming lessons from city radio in york",
    "alexa browse nutrition tips near exeter",
    "alexa start meditation from health hour in norwich",
    "alexa find yoga classes near bath",
    "alexa listen to wellness from south london radio",
    "alexa hear climate change news in portsmouth",
    "alexa show me space from tech today near plymouth",
    "alexa browse robotics from james miller in leicester",
    "alexa play the latest film reviews",
    "alexa listen to education from alan ross near cardiff",
    "alexa hear politics news in belfast",
    "alexa show me business from city press near london",
    "alexa browse weather forecast in manchester",
    "alexa play baseball from global sports desk near birmingham",
    "alexa listen to dance music from sarah near glasgow",
    "alexa hear animation from emily clarke",
    "alexa show me history documentaries from robert king in liverpool",
    "alexa browse cooking shows from lisa adams near bristol",
    "alexa play travel guides from tom baker",
    "alexa listen to photography tips from mark williams in edinburgh",
    "alexa hear religion from james miller near newcastle",
    "alexa show me law from city radio in cardiff",
    "alexa browse economics from northern report near belfast",
    "alexa play football from the podcast network in london",
]


async def send_request(client: httpx.AsyncClient, utterance: str, idx: int) -> dict:
    start = time.perf_counter()
    try:
        resp = await client.post(URL, json={"utterance": utterance, "country_code": "gb"}, timeout=15.0)
        elapsed = (time.perf_counter() - start) * 1000
        data = resp.json()
        cat = data.get("category") or {}
        cre = data.get("creator") or {}
        loc = data.get("location") or {}
        return {"idx": idx, "status": resp.status_code, "elapsed_ms": elapsed, "category": cat.get("slug"), "creator": cre.get("name"), "location": loc.get("city"), "error": None}
    except Exception as e:
        elapsed = (time.perf_counter() - start) * 1000
        return {"idx": idx, "status": 0, "elapsed_ms": elapsed, "category": None, "creator": None, "location": None, "error": str(e)[:100]}


async def run_load_test():
    utterances = list(TEMPLATES) + list(ALEXA_STYLE)
    print(f"Target: {URL}")
    print(f"Concurrent: {CONCURRENT} | Utterances: {len(utterances)}")
    print(f"{'='*70}")

    async with httpx.AsyncClient(limits=httpx.Limits(max_connections=CONCURRENT, max_keepalive_connections=100)) as client:
        start_all = time.perf_counter()
        tasks = [send_request(client, random.choice(utterances), i) for i in range(CONCURRENT)]
        results = await asyncio.gather(*tasks)
        total_elapsed = time.perf_counter() - start_all

    errors = sum(1 for r in results if r["error"])
    ok = sum(1 for r in results if 200 <= r["status"] < 300)
    not_ok = CONCURRENT - ok - errors
    timings = sorted([r["elapsed_ms"] for r in results])
    n = len(timings)
    avg = sum(timings) / n

    print(f"Total: {total_elapsed:.1f}s for {CONCURRENT} requests")
    print(f"OK: {ok} | Not OK: {not_ok} | Errors: {errors}")
    print(f"QPS: {round(CONCURRENT/total_elapsed, 0)}")
    print(f"\nLatency (ms):")
    print(f"  avg: {round(avg, 1)}")
    print(f"  p50: {round(timings[int(n*0.5)], 1)}")
    print(f"  p90: {round(timings[int(n*0.9)], 1)}")
    print(f"  p95: {round(timings[int(n*0.95)], 1)}")
    print(f"  p99: {round(timings[int(n*0.99)], 1)}")
    print(f"  max: {round(timings[-1], 1)}")
    print(f"  min: {round(timings[0], 1)}")

    if errors:
        print(f"\nErrors ({errors}):")
        for r in results:
            if r["error"]:
                print(f"  #{r['idx']}: {r['error'][:80]}")

    resolved = sum(1 for r in results if r["category"] or r["creator"] or r["location"])
    print(f"\nResolved: {resolved}/{CONCURRENT} ({round(resolved/CONCURRENT*100, 1)}%)")

    print(f"\nSample responses:")
    for r in results[:10]:
        print(f"  #{r['idx']:3d} {r['elapsed_ms']:6.0f}ms | cat={r['category'] or '-':15s} creator={r['creator'] or '-':20s} loc={r['location'] or '-'}")

asyncio.run(run_load_test())
