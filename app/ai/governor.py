"""Client-side rate limiting.

Groq's Developer plan allows 250K tokens and 1K requests per minute per model.
The extraction fan-out (one structured call per chunk window, run in parallel)
is what reaches that first, and the failure mode without a ceiling is a 429
storm in the middle of a pipeline run -- retried requests spending the very
budget that is exhausted.

So the ceiling is enforced here, before the request leaves. Three limits, all
of which must be satisfied:

*   **concurrency** -- a plain semaphore, because Groq's per-request latency is
    low and unbounded fan-out buys nothing.
*   **tokens per minute** -- a leaky bucket refilled continuously rather than
    reset on a minute boundary. A boundary reset lets a burst spend the whole
    minute's budget in one second, which is exactly the shape that 429s.
*   **requests per minute** -- the same bucket, counting requests.

Token cost has to be *estimated* before the call, since Groq exposes no
token-counting endpoint. The estimate is deliberately crude and deliberately
high; ``reconcile`` corrects the bucket afterwards from the usage the response
actually reports, so a persistent bias self-cancels within a few requests
instead of accumulating.

No LangChain import here on purpose: this module is the part of the AI layer
that must be unit-testable on any interpreter, with no provider installed.
"""

import asyncio
import time
from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional


class _LeakyBucket:
    """A bucket of `capacity` units that refills at `capacity / period` per second."""

    def __init__(self, capacity: float, period_seconds: float = 60.0) -> None:
        self._capacity = float(capacity)
        self._rate = float(capacity) / float(period_seconds)
        self._level = float(capacity)
        self._updated_at = time.monotonic()
        self._lock = asyncio.Lock()

    def _refill(self) -> None:
        now = time.monotonic()
        self._level = min(self._capacity, self._level + (now - self._updated_at) * self._rate)
        self._updated_at = now

    async def take(self, amount: float) -> None:
        """Block until `amount` units are available, then remove them.

        A request larger than the whole bucket would wait forever, so it is
        clamped: one oversized request is throttled to a full bucket's worth
        rather than deadlocking the worker.
        """
        amount = min(float(amount), self._capacity)
        while True:
            async with self._lock:
                self._refill()
                if self._level >= amount:
                    self._level -= amount
                    return
                deficit = amount - self._level
                wait = deficit / self._rate if self._rate > 0 else 0.1
            await asyncio.sleep(max(wait, 0.01))

    async def give_back(self, amount: float) -> None:
        async with self._lock:
            self._refill()
            self._level = min(self._capacity, self._level + float(amount))

    @property
    def level(self) -> float:
        self._refill()
        return self._level


def estimate_tokens(text: str) -> int:
    """A deliberately rough upper bound: ~4 characters per token, plus overhead.

    Only the bucket reads this, and `reconcile` corrects it from real usage
    afterwards. Precision here would cost a tokenizer dependency and buy
    nothing -- being consistently a little high is the safe direction.
    """
    return int(len(text) / 3.5) + 64


class RateLimitGovernor:
    """One instance per process, shared by every task."""

    def __init__(
        self,
        max_concurrency: int,
        tokens_per_minute: int,
        requests_per_minute: int,
    ) -> None:
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._tokens = _LeakyBucket(tokens_per_minute)
        self._requests = _LeakyBucket(requests_per_minute)

    @asynccontextmanager
    async def reserve(self, estimated_tokens: int) -> AsyncIterator[None]:
        """Hold a slot for one request.

        Order matters: take from the buckets *before* the semaphore. Holding a
        concurrency slot while waiting on a bucket keeps other callers out for
        no reason.
        """
        await self._requests.take(1)
        await self._tokens.take(estimated_tokens)
        async with self._semaphore:
            yield

    async def reconcile(self, estimated_tokens: int, actual_tokens: Optional[int]) -> None:
        """Correct the token bucket once the response reports real usage."""
        if actual_tokens is None:
            return
        drift = estimated_tokens - actual_tokens
        if drift > 0:
            await self._tokens.give_back(drift)
        elif drift < 0:
            await self._tokens.take(-drift)

    @property
    def tokens_available(self) -> float:
        return self._tokens.level
