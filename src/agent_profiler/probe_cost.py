"""Account for invocations independently of whether they yield a measurement."""

from __future__ import annotations


class ProbeCost:
    def __init__(self):
        self.calls = 0
        self.failed = 0
        self.known = 0.0
        self.unknown = 0

    def record(self, result: dict) -> None:
        self.calls += 1
        self.failed += not result.get("ok", False)
        cost = result.get("cost_usd")
        if cost is None:
            self.unknown += 1
        else:
            self.known += cost

    def include(self, result: dict) -> None:
        """Merge another component's accounting, including failed components."""
        self.calls += result.get("llm_calls", 0)
        self.failed += result.get("failed_calls", 0)
        self.known += result.get("known_cost_usd", 0.0)
        self.unknown += result.get("unknown_cost_calls", 0)

    def summary(self) -> dict:
        return {
            "llm_calls": self.calls,
            "failed_calls": self.failed,
            "known_cost_usd": round(self.known, 6),
            "unknown_cost_calls": self.unknown,
            "cost_usd": None if self.unknown else round(self.known, 6),
        }
