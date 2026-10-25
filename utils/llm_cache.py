"""
utils/llm_cache.py
==================
Disk cache, rate limiting and retry/backoff for hosted-LLM calls.

Evaluation runs issue thousands of Groq calls against a free tier, so every
call goes through ``cached_call``: identical requests (same model, prompt and
image bytes) are answered from disk, interrupted runs resume where they
stopped, and ablations that only change post-processing re-use stored model
output instead of spending quota.

Environment variables
---------------------
LLM_CACHE_DIR   cache directory (default: data/cache/llm)
LLM_CACHE       set to "0" to bypass the cache entirely
GROQ_RPM        max Groq requests per minute (default: 20)
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

_CACHE_DIR = Path(os.getenv("LLM_CACHE_DIR", Path("data") / "cache" / "llm"))
_lock = threading.Lock()
_stats = {"hits": 0, "misses": 0}


def cache_enabled() -> bool:
    return os.getenv("LLM_CACHE", "1") != "0"


def cache_stats() -> dict:
    return dict(_stats)


_usage: dict = {}


def record_usage(provider: str, calls: int = 1, prompt_tokens: int = 0, completion_tokens: int = 0) -> None:
    """Count real (uncached) LLM calls and tokens, e.g. provider="groq" or "ollama"."""
    with _lock:
        for key, inc in (("calls", calls), ("prompt_tokens", prompt_tokens),
                         ("completion_tokens", completion_tokens)):
            name = f"{provider}_{key}"
            _usage[name] = _usage.get(name, 0) + int(inc or 0)


def usage_snapshot() -> dict:
    with _lock:
        return dict(_usage)


def usage_delta(before: dict) -> dict:
    after = usage_snapshot()
    return {k: after.get(k, 0) - before.get(k, 0) for k in set(after) | set(before)}


def make_key(*parts: Any) -> str:
    """Stable SHA-256 over strings, bytes and JSON-serialisable values."""
    h = hashlib.sha256()
    for part in parts:
        if isinstance(part, bytes):
            h.update(part)
        elif isinstance(part, str):
            h.update(part.encode("utf-8"))
        else:
            h.update(json.dumps(part, sort_keys=True, default=str).encode("utf-8"))
        h.update(b"\x1f")
    return h.hexdigest()


def _path(namespace: str, key: str) -> Path:
    return _CACHE_DIR / namespace / key[:2] / f"{key}.json"


def cache_get(namespace: str, key: str) -> Optional[Any]:
    if not cache_enabled():
        return None
    p = _path(namespace, key)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))["value"]
    except (OSError, ValueError, KeyError):
        return None


def cache_put(namespace: str, key: str, value: Any) -> None:
    if not cache_enabled():
        return
    p = _path(namespace, key)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps({"value": value}, ensure_ascii=False), encoding="utf-8")
    tmp.replace(p)


class RateLimiter:
    """Spaces calls at least ``60 / rpm`` seconds apart across threads."""

    def __init__(self, rpm: float):
        self.interval = 60.0 / rpm if rpm > 0 else 0.0
        self._next = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        if not self.interval:
            return
        with self._lock:
            now = time.monotonic()
            delay = max(0.0, self._next - now)
            self._next = max(now, self._next) + self.interval
        if delay:
            time.sleep(delay)


_groq_limiter = RateLimiter(float(os.getenv("GROQ_RPM", "20")))


def groq_throttle() -> None:
    _groq_limiter.wait()


def _retry_after(exc: Exception) -> Optional[float]:
    headers = getattr(getattr(exc, "response", None), "headers", None)
    if headers:
        try:
            return float(headers.get("retry-after"))
        except (TypeError, ValueError):
            return None
    return None


def with_backoff(fn: Callable[[], Any], retries: int = 6, base: float = 2.0) -> Any:
    """Retry rate-limit and transient errors with exponential backoff."""
    for attempt in range(retries):
        try:
            return fn()
        except Exception as exc:
            text = str(exc).lower()
            transient = any(t in text for t in ("429", "rate limit", "timeout", "503", "502", "overloaded"))
            if not transient or attempt == retries - 1:
                raise
            delay = _retry_after(exc) or base * (2 ** attempt) + random.random()
            logger.warning("LLM call failed (%s); retrying in %.1fs", exc, delay)
            time.sleep(delay)


def cached_call(namespace: str, key: str, fn: Callable[[], Any], *, groq: bool = False) -> Any:
    """
    Return the cached value for ``key`` or compute it with ``fn``.

    ``fn`` is retried with backoff on rate-limit errors and, when ``groq`` is
    set, throttled to ``GROQ_RPM``. Results that are ``None`` are not cached.
    """
    hit = cache_get(namespace, key)
    if hit is not None:
        with _lock:
            _stats["hits"] += 1
        return hit

    def _run():
        if groq:
            groq_throttle()
        return fn()

    value = with_backoff(_run)
    with _lock:
        _stats["misses"] += 1
    if value is not None:
        cache_put(namespace, key, value)
    return value
