import re
from datetime import datetime, timedelta, timezone

_WORD_NUMS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}

_WEEKDAYS = {
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
    "friday": 4, "saturday": 5, "sunday": 6,
}

_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
}

_RECENCY = {"latest", "newest", "recent", "recently", "new", "current"}
_TIME_OF_DAY = {"morning", "afternoon", "evening", "tonight", "night"}


def _iso(d: datetime) -> str:
    return d.date().isoformat()


def detect_temporal(utterance: str) -> dict | None:
    text = utterance.lower()
    words = text.split()
    now = datetime.now(timezone.utc)

    if "yesterday" in words:
        return {"type": "date", "value": "yesterday", "date": _iso(now - timedelta(days=1))}
    if "today" in words:
        return {"type": "date", "value": "today", "date": _iso(now)}
    if "tomorrow" in words:
        return {"type": "date", "value": "tomorrow", "date": _iso(now + timedelta(days=1))}

    if re.search(r"\blasts?\s+weekend\b", text):
        return {"type": "date", "value": "last_weekend", "date": _iso(now - timedelta(days=(now.weekday() + 2) % 7 or 7))}
    if re.search(r"\bthis\s+weekend\b", text):
        return {"type": "date", "value": "this_weekend", "date": _iso(now + timedelta(days=(5 - now.weekday()) % 7 or 7))}
    if re.search(r"\blast\s+week\b", text):
        return {"type": "range", "value": "last_week", "date": _iso(now - timedelta(days=7))}
    if re.search(r"\bthis\s+week\b", text):
        return {"type": "range", "value": "this_week", "date": None}
    if re.search(r"\blast\s+month\b", text):
        return {"type": "range", "value": "last_month", "date": None}
    if re.search(r"\bthis\s+month\b", text):
        return {"type": "range", "value": "this_month", "date": None}

    if "last" in words:
        i = words.index("last")
        if i + 1 < len(words) and words[i + 1] in _WEEKDAYS:
            target = _WEEKDAYS[words[i + 1]]
            delta = (now.weekday() - target) % 7 or 7
            return {"type": "date", "value": f"last_{words[i + 1]}", "date": _iso(now - timedelta(days=delta))}

    match = re.search(r"(\d+)\s+(day|days|week|weeks|month|months)\s+ago", text)
    if match:
        num = int(match.group(1))
        unit = match.group(2)
        if unit in ("day", "days"):
            d = now - timedelta(days=num)
            return {"type": "date", "value": f"{num}_days_ago", "date": _iso(d)}
        elif unit in ("week", "weeks"):
            d = now - timedelta(weeks=num)
            return {"type": "date", "value": f"{num}_weeks_ago", "date": _iso(d)}
        elif unit in ("month", "months"):
            d = now - timedelta(days=num * 30)
            return {"type": "date", "value": f"{num}_months_ago", "date": _iso(d)}
    match = re.search(r"(one|two|three|four|five|six|seven|eight|nine|ten)\s+(day|days|week|weeks|month|months)\s+ago", text)
    if match:
        num = _WORD_NUMS[match.group(1)]
        unit = match.group(2)
        if unit in ("day", "days"):
            d = now - timedelta(days=num)
            return {"type": "date", "value": f"{num}_days_ago", "date": _iso(d)}
        elif unit in ("week", "weeks"):
            d = now - timedelta(weeks=num)
            return {"type": "date", "value": f"{num}_weeks_ago", "date": _iso(d)}
        elif unit in ("month", "months"):
            d = now - timedelta(days=num * 30)
            return {"type": "date", "value": f"{num}_months_ago", "date": _iso(d)}

    match = re.search(
        r"(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d+)(?:st|nd|rd|th)?",
        text,
    )
    if match:
        month_name = match.group(1)
        day = int(match.group(2))
        month = _MONTHS[month_name]
        year = now.year
        try:
            d = datetime(year, month, day, tzinfo=timezone.utc)
            return {"type": "date", "value": f"{month_name}_{day}", "date": _iso(d)}
        except ValueError:
            pass

    if "last night" in text:
        return {"type": "date", "value": "last_night", "date": _iso(now - timedelta(days=1))}

    for w in words:
        if w in _RECENCY:
            return {"type": "recency", "value": "latest", "date": None}

    for i, w in enumerate(words):
        if w in _WEEKDAYS:
            if i > 0 and words[i - 1] in ("this", "on"):
                target = _WEEKDAYS[w]
                delta = (target - now.weekday()) % 7
                if delta == 0:
                    delta = 0 if words[i - 1] == "this" else 7
                d = now + timedelta(days=delta)
                return {"type": "date", "value": f"{words[i-1]}_{w}", "date": _iso(d)}

    for tod in _TIME_OF_DAY:
        if tod in words:
            return {"type": "time_of_day", "value": tod, "date": None}

    return None
