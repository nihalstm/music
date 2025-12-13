import csv
from pathlib import Path

def load_alias_map(csv_path: str) -> dict[str, str]:
    amap = {}
    p = Path(csv_path)
    with p.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            a = (row["alias"] or "").strip().lower()
            c = (row["canonical"] or "").strip()
            if a and c:
                amap[a] = c
    # longer aliases first for substring matching
    return dict(sorted(amap.items(), key=lambda kv: len(kv[0]), reverse=True))

def apply_alias_boosts(text: str, scores: dict[str, float], alias_map: dict[str,str], delta: float = 0.5) -> dict[str, float]:
    # add small logit bump to matched canonical labels
    t = (text or "").lower()
    boosted = dict(scores)
    for alias, canonical in alias_map.items():
        if alias in t:
            boosted[canonical] = boosted.get(canonical, 0.0) + float(delta)
    return boosted
