"""
backend/ml/active_learning.py

Dynamic dataset helpers:
- Log every inference to UNLABELED
- Auto-promote confident items to SILVER (strict thresholds)
- Record human feedback as GOLD
- Basic near-dup guard and SILVER invalidation on GOLD

All I/O is best-effort; callers should never fail because logging failed.
"""

from __future__ import annotations

import os
import json
import time
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
from simhash import Simhash

# -------------------- Config --------------------

# Promotion thresholds (env-overridable)
THRESH_P1 = float(os.getenv("AL_PROMOTE_MIN_P1", "0.90"))
THRESH_MARGIN = float(os.getenv("AL_PROMOTE_MIN_MARGIN", "0.35"))
# Optional logging gate: skip logging if top2 gap is tiny
MIN_LOG_GAP = float(os.getenv("AL_MIN_LOG_GAP", "0.05"))

# Terms that should NOT promote certain labels
CONFLICT_TERMS: Dict[str, set[str]] = {
    "Epic": {"headache", "dizzy", "overheated", "burned", "tilted", "stress"},
    "Warm": {"sleep", "sleepy", "tired", "drained", "exhausted", "nap", "lie down", "close my eyes", "recover", "quiet"},
    "Bittersweet": {"sleep", "sleepy", "tired", "drained", "exhausted", "nap", "lie down", "close my eyes", "recover", "quiet"},
}

# Terms that MAY promote only a small set of labels; anything else is suspicious
ALLOW_SILVER: Dict[str, set[str]] = {
    "sleep": {"Calm", "Mellow", "Ethereal"},
    "sleepy": {"Calm", "Mellow", "Ethereal"},
    "tired": {"Calm", "Tense", "Melancholic"},
    "drained": {"Calm", "Tense", "Melancholic"},
    "exhausted": {"Calm", "Tense", "Melancholic"},
    "nap": {"Calm", "Mellow"},
    "lie down": {"Calm", "Mellow", "Ethereal"},
    "close my eyes": {"Calm", "Mellow", "Ethereal"},
    "recover": {"Calm"},
    "quiet": {"Calm", "Mellow"},
}

# -------------------- Path safety helpers --------------------

def _canonical_data_dir() -> Path:
    """Canonical repo data dir (<repo>/data)."""
    root = Path(__file__).resolve().parents[2]
    return Path(os.getenv("MUSIC_DATA_DIR", str(root / "data")))

def _normalize_path(path: str) -> Path:
    """Ensure any 'data/foo.jsonl' or 'foo.jsonl' resolves to <repo>/data/foo.jsonl."""
    p = Path(path)
    if not p.is_absolute():
        name = p.name
        p = _canonical_data_dir() / name
    return p

# -------------------- Utils --------------------

def _append_jsonl(path: str, row: dict) -> None:
    """Append a single JSON object to a JSONL file (best-effort, CWD-proof)."""
    try:
        p = _normalize_path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"[active_learning] warn: append failed for {path}: {e}")

def _read_jsonl(path: str) -> list[dict]:
    try:
        p = _normalize_path(path)
        with p.open("r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    except FileNotFoundError:
        return []
    except Exception as e:
        print(f"[active_learning] warn: read failed for {path}: {e}")
        return []

def text_fingerprint(text: str) -> str:
    """Stable, fast near-duplicate fingerprint."""
    return str(Simhash((text or "").lower()).value)

def entropy_from_probs(probs: Iterable[float]) -> float:
    """Shannon entropy (nats) for a probability vector."""
    p = np.clip(np.array(list(probs), dtype=float), 1e-9, 1.0)
    return float(-(p * np.log(p)).sum())

def _rank_scores(scores: Dict[str, float]) -> list[Tuple[str, float]]:
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)

def _has_conflict(text: str, label: str) -> bool:
    terms = CONFLICT_TERMS.get(label)
    if not terms:
        return False
    t = (text or "").lower()
    return any(term in t for term in terms)

def _violates_allowlist(text: str, label: str) -> bool:
    t = (text or "").lower()
    for term, allowed in ALLOW_SILVER.items():
        if term in t and label not in allowed:
            return True
    return False

# -------------------- Public API --------------------

def promote_silver(scores: Dict[str, float], text: str) -> bool:
    """
    Auto-promote to SILVER if:
      - top1 prob >= THRESH_P1
      - (top1 - top2) >= THRESH_MARGIN
      - no conflict keywords for that label
      - does not violate allowlist for detected terms
    """
    ranked = _rank_scores(scores)
    (l1, p1) = ranked[0]
    (l2, p2) = ranked[1] if len(ranked) > 1 else ("__none__", 0.0)

    if p1 < THRESH_P1:
        return False
    if (p1 - p2) < THRESH_MARGIN:
        return False
    if _has_conflict(text, l1):
        return False
    if _violates_allowlist(text, l1):
        return False
    return True

def log_unlabeled(path_unlabeled: str, text: str, scores: Dict[str, float], model_version: str) -> None:
    """Log raw inference with uncertainty + margin. Optional margin skim."""
    ranked = _rank_scores(scores)
    pvec = [p for _, p in ranked]
    row = {
        "text": text,
        "scores": dict(ranked),
        "top1": ranked[0][0],
        "top2": ranked[1][0] if len(ranked) > 1 else None,
        "uncertainty": entropy_from_probs(pvec),
        "margin": float(pvec[0] - (pvec[1] if len(pvec) > 1 else 0.0)),
        "fp": text_fingerprint(text),
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model_version": model_version,
        "source": "inference",
    }
    if MIN_LOG_GAP > 0 and row["margin"] < MIN_LOG_GAP:
        return
    _append_jsonl(path_unlabeled, row)

def log_silver(path_silver: str, text: str, scores: Dict[str, float], model_version: str) -> None:
    """Write a high-confidence auto-labeled example."""
    ranked = _rank_scores(scores)
    row = {
        "text": text,
        "labels": [ranked[0][0]],
        "rejected": [],
        "scores": dict(ranked),
        "weight": 0.3,
        "fp": text_fingerprint(text),
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model_version": model_version,
        "source": "silver",
    }
    _append_jsonl(path_silver, row)

def log_gold(
    path_gold: str,
    text: str,
    chosen: list[str],
    rejected: Optional[list[str]] = None,
    user_id_hash: Optional[str] = None,
) -> None:
    """Record human-confirmed labels (GOLD)."""
    row = {
        "text": text,
        "labels": list(chosen or []),
        "rejected": list(rejected or []),
        "weight": 1.0,
        "fp": text_fingerprint(text),
        "user_id_hash": user_id_hash,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": "user_feedback",
    }
    _append_jsonl(path_gold, row)

# -------------------- Maintenance helpers (optional) --------------------

def invalidate_silver_for_fp(path_silver: str, fp: str) -> int:
    """Remove SILVER rows that match a given fingerprint. Returns count removed."""
    rows = _read_jsonl(path_silver)
    if not rows:
        return 0
    keep, removed = [], 0
    for r in rows:
        if str(r.get("fp")) == str(fp):
            removed += 1
        else:
            keep.append(r)
    try:
        p = _normalize_path(path_silver)
        tmp = p.with_suffix(p.suffix + ".tmp")
        p.parent.mkdir(parents=True, exist_ok=True)
        with tmp.open("w", encoding="utf-8") as f:
            for r in keep:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        os.replace(tmp, p)
    except Exception as e:
        print(f"[active_learning] warn: invalidate_silver_for_fp failed: {e}")
        return 0
    return removed
