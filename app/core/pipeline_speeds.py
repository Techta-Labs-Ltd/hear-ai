def parse_speed_multiplier_csv(raw: str) -> list[float]:
    out: list[float] = []
    for part in (raw or "").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            v = float(part)
        except ValueError:
            continue
        if 0.5 <= v <= 3.0 and abs(v - 1.0) > 1e-6:
            out.append(round(v, 4))
    return sorted(set(out))


def merge_speed_multipliers(
    defaults: list[float],
    job_speeds: list[float] | None,
    instruction_speeds: list[float] | None,
) -> list[float]:
    merged: set[float] = set(defaults)
    for src in (job_speeds or [], instruction_speeds or []):
        for x in src:
            try:
                v = float(x)
            except (TypeError, ValueError):
                continue
            if 0.5 <= v <= 3.0 and abs(v - 1.0) > 1e-6:
                merged.add(round(v, 4))
    return sorted(merged)
