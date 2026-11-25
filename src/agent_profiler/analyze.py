"""Deterministic analyzer.

Stage 4 turns normalized events into findings. It reads **events only** — never a
transcript — which is what Stage 3 exists to make possible. Zero LLM tokens.

Two rules govern everything here:

1. **A finding's strength never exceeds its evidence.** Every finding carries a
   `strength` of `observed` or `estimated`. `observed` means the events prove it;
   `estimated` means a stated assumption was applied and the finding says which.
   Nothing in this module produces a counterfactual claim ("removing X would save Y"):
   that needs a real A/B, which is Stage 6.

2. **A confound is reported, not resolved by guessing.** The dogfood that drove this
   module found that *every* repeated identical call across a real corpus had an opaque
   command (Bash/exec) between the repeats, which can change the file being re-read.
   Such repeats are therefore classified `inconclusive` rather than counted as waste.
.
"""

from __future__ import annotations

import time
from collections import Counter, defaultdict
from pathlib import Path

from .events import EVENT_TYPES, normalize_root, normalize_session

ANALYZER_SCHEMA_VERSION = "0.4"

# Tools whose effect on the filesystem is not visible in the event stream. A repeat
# separated by one of these cannot be called avoidable: the target may have changed.
OPAQUE_TOOLS = frozenset(
    {"Bash", "BashOutput", "exec", "js", "write_stdin", "Task", "KillShell", "sleep"}
)
# Tools that write to a target, making a later re-read of that target legitimate.
MUTATING_TOOLS = frozenset({"Edit", "Write", "NotebookEdit", "apply_patch", "MultiEdit"})

# A cache hit rate below this is worth a look, but it is strongly driven by how long a
# session ran, so the finding says so instead of implying waste.
LOW_CACHE_HIT_PCT = 70.0
# Sessions shorter than this have too few responses for a hit rate to mean anything.
MIN_RESPONSES_FOR_CACHE_FINDING = 8
# Reporting a failure rate on a handful of calls is noise.
MIN_CALLS_FOR_ERROR_RATE = 20


def _finding(
    kind: str,
    *,
    strength: str,
    summary: str,
    evidence: dict,
    scope: str = "run",
    assumptions: list[str] | None = None,
    judgment: str = "review_only",
    references: list[dict] | None = None,
) -> dict:
    assert strength in {"observed", "estimated"}, strength
    finding = {
        "kind": kind,
        "strength": strength,
        "scope": scope,
        "summary": summary,
        "evidence": evidence,
        "judgment": judgment,
    }
    if assumptions:
        finding["assumptions"] = assumptions
    if references:
        finding["references"] = references[:10]
    return finding


def _classify_repeats(events: list[dict]) -> dict:
    """Group identical calls and say, per repeat, whether the events can explain it.

    A repeat is only `avoidable_candidate` when nothing between the two calls could have
    changed the result: no opaque command, no write to the same target, no compaction,
    and the earlier call did not fail.
    """
    calls = [e for e in events if e["type"] == "tool_call"]
    results = {e.get("call_id"): e for e in events if e["type"] == "tool_result"}
    compactions = [e["seq"] for e in events if e["type"] == "context_compaction"]

    by_signature: dict[tuple, list[dict]] = defaultdict(list)
    for call in calls:
        by_signature[(call.get("tool"), call.get("input_digest"))].append(call)

    buckets: Counter = Counter()
    groups = []
    for (tool, digest), hits in by_signature.items():
        if len(hits) < 2:
            continue
        classifications = []
        for first, second in zip(hits, hits[1:]):
            between = [c for c in calls if first["seq"] < c["seq"] < second["seq"]]
            prior = results.get(first.get("call_id"))
            target = first.get("target")
            reasons = []
            if prior is not None and prior.get("error"):
                reasons.append("earlier_call_failed")
            if any(c.get("tool") in OPAQUE_TOOLS for c in between):
                reasons.append("opaque_command_between")
            if target and any(
                c.get("tool") in MUTATING_TOOLS and c.get("target") == target for c in between
            ):
                reasons.append("target_written_between")
            if any(first["seq"] < c < second["seq"] for c in compactions):
                reasons.append("context_compacted_between")

            if "earlier_call_failed" in reasons:
                verdict = "retry_after_failure"
            elif reasons:
                verdict = "inconclusive"
            else:
                verdict = "avoidable_candidate"
            buckets[verdict] += 1
            classifications.append({"verdict": verdict, "reasons": reasons, "seqs": [first["seq"], second["seq"]]})

        groups.append(
            {
                "tool": tool,
                "input_digest": digest,
                "target": hits[0].get("target"),
                "calls": len(hits),
                "seqs": [h["seq"] for h in hits],
                "repeats": classifications,
            }
        )

    groups.sort(key=lambda g: -g["calls"])
    return {"groups": groups, "verdicts": dict(sorted(buckets.items()))}


def _token_profile(events: list[dict]) -> dict:
    responses = [e for e in events if e["type"] == "model_response"]
    totals: Counter = Counter()
    for response in responses:
        for key, value in (response.get("tokens") or {}).items():
            if isinstance(value, int):
                totals[key] += value

    input_total = totals["input_total"]
    cache_hit_pct = round(totals["input_cached"] / input_total * 100, 2) if input_total else None
    largest = max(responses, key=lambda e: (e.get("tokens") or {}).get("input_total") or 0, default=None)
    return {
        "responses": len(responses),
        "interrupted_responses": sum(1 for e in responses if e.get("incomplete")),
        "tokens": dict(sorted(totals.items())),
        "cache_hit_pct": cache_hit_pct,
        "largest_response_input": (largest.get("tokens") or {}).get("input_total") if largest else None,
    }


def _tool_profile(events: list[dict]) -> dict:
    calls = Counter()
    errors = Counter()
    error_refs = []
    for event in events:
        if event["type"] == "tool_call":
            calls[event.get("tool") or "__unknown__"] += 1
        elif event["type"] == "tool_result" and event.get("error"):
            errors[event.get("tool") or "__unattributed__"] += 1
            if len(error_refs) < 10:
                error_refs.append({"seq": event["seq"], "tool": event.get("tool"), "raw": event.get("raw")})
    total = sum(calls.values())
    return {
        "calls_total": total,
        "errors_total": sum(errors.values()),
        "error_rate_pct": round(sum(errors.values()) / total * 100, 2) if total else None,
        "calls_by_tool": dict(calls.most_common()),
        "errors_by_tool": dict(errors.most_common()),
        "error_references": error_refs,
    }


def _analyze_events(events: list[dict], *, scope: str) -> tuple[list[dict], dict]:
    """Produce findings plus the measurements they rest on."""
    findings: list[dict] = []
    repeats = _classify_repeats(events)
    tokens = _token_profile(events)
    tools = _tool_profile(events)

    avoidable = [
        {"tool": g["tool"], "target": g["target"], "seqs": r["seqs"]}
        for g in repeats["groups"]
        for r in g["repeats"]
        if r["verdict"] == "avoidable_candidate"
    ]
    if avoidable:
        findings.append(
            _finding(
                "repeated_identical_call",
                strength="observed",
                scope=scope,
                summary=f"{len(avoidable)} identical tool call(s) repeated with nothing in between that "
                "could have changed the result.",
                evidence={
                    "repeats": len(avoidable),
                    "detail": avoidable[:10],
                    "ruled_out": ["opaque_command_between", "target_written_between",
                                  "context_compacted_between", "earlier_call_failed"],
                },
            )
        )

    inconclusive = repeats["verdicts"].get("inconclusive", 0)
    if inconclusive:
        reason_counts: Counter = Counter()
        for group in repeats["groups"]:
            for repeat in group["repeats"]:
                if repeat["verdict"] == "inconclusive":
                    reason_counts.update(repeat["reasons"])
        findings.append(
            _finding(
                "repeated_call_inconclusive",
                strength="observed",
                scope=scope,
                summary=f"{inconclusive} repeated identical call(s) cannot be judged: something between "
                "the calls may legitimately have changed the result.",
                evidence={"repeats": inconclusive, "reasons": dict(reason_counts.most_common())},
                judgment="not_waste",
            )
        )

    retries = repeats["verdicts"].get("retry_after_failure", 0)
    if retries:
        findings.append(
            _finding(
                "retry_after_failure",
                strength="observed",
                scope=scope,
                summary=f"{retries} call(s) were repeated after the identical earlier call returned an error.",
                evidence={"repeats": retries},
            )
        )

    if tools["calls_total"] >= MIN_CALLS_FOR_ERROR_RATE and tools["errors_total"]:
        findings.append(
            _finding(
                "tool_failures",
                strength="observed",
                scope=scope,
                summary=f"{tools['errors_total']} of {tools['calls_total']} tool calls returned an error "
                f"({tools['error_rate_pct']}%).",
                evidence={
                    "errors": tools["errors_total"],
                    "calls": tools["calls_total"],
                    "rate_pct": tools["error_rate_pct"],
                    "by_tool": tools["errors_by_tool"],
                },
                references=tools["error_references"],
            )
        )

    hit = tokens["cache_hit_pct"]
    if (
        hit is not None
        and hit < LOW_CACHE_HIT_PCT
        and tokens["responses"] >= MIN_RESPONSES_FOR_CACHE_FINDING
    ):
        findings.append(
            _finding(
                "low_cache_hit_rate",
                strength="estimated",
                scope=scope,
                summary=f"Cached input was {hit}% of total input across {tokens['responses']} responses.",
                evidence={
                    "cache_hit_pct": hit,
                    "responses": tokens["responses"],
                    "input_total": tokens["tokens"].get("input_total"),
                    "input_cache_write": tokens["tokens"].get("input_cache_write"),
                },
                assumptions=[
                    "Cache hit rate rises with session length because later responses reuse a larger "
                    "prefix, so a low rate on a short session is expected rather than wasteful.",
                    "No claim is made that a higher hit rate was achievable in this run.",
                ],
            )
        )

    if tokens["interrupted_responses"]:
        findings.append(
            _finding(
                "interrupted_responses",
                strength="observed",
                scope=scope,
                summary=f"{tokens['interrupted_responses']} response(s) ended without a stop reason.",
                evidence={"responses": tokens["interrupted_responses"]},
                judgment="review_only",
            )
        )

    return findings, {"tokens": tokens, "tools": tools, "repeats": repeats}


def _usage_findings(scan: dict, used_tools: Counter, harnesses: set[str], sessions: int) -> list[dict]:
    """Bridge Installed -> Used, refusing to claim more than the traces can support.

    Declared-but-never-called is only reportable for a harness whose traces were actually
    analyzed. A Codex-only MCP server is not "unused" because Claude Code sessions never
    called it — it was never available to them.
    """
    findings: list[dict] = []
    servers = scan.get("mcp_servers") or []
    if not servers:
        return findings

    # source -> which harness can load it
    harness_of_source = {
        "project_root": "claude_code",
        "user_claude_json": "claude_code",
        "project_claude_dir": "claude_code",
        "user_claude_dir": "claude_code",
        "project_codex_dir": "codex",
        "user_codex_dir": "codex",
    }
    called = Counter()
    for tool, count in used_tools.items():
        if tool.startswith("mcp__"):
            parts = tool.split("__")
            if len(parts) > 1:
                called[parts[1]] += count

    by_name: dict[str, set[str]] = defaultdict(set)
    for server in servers:
        mapped = harness_of_source.get(server.get("source"))
        if mapped:
            by_name[server["name"]].add(mapped)

    unused = []
    unverifiable = []
    for name, declaring in sorted(by_name.items()):
        if called.get(name):
            continue
        if declaring & harnesses:
            unused.append({"name": name, "declared_for": sorted(declaring)})
        else:
            unverifiable.append({"name": name, "declared_for": sorted(declaring)})

    if unused:
        findings.append(
            _finding(
                "declared_mcp_never_called",
                strength="observed",
                scope="root",
                summary=f"{len(unused)} MCP server(s) declared for an analyzed harness were never called "
                f"in {sessions} session(s).",
                evidence={"servers": unused, "sessions_analyzed": sessions,
                          "harnesses_analyzed": sorted(harnesses)},
                assumptions=None,
                judgment="review_only",
            )
        )
        findings[-1]["limits"] = (
            "Never called in the analyzed sessions. That is not evidence of low value: the server "
            "still occupies tool-schema context in every session, and it may be needed for work "
            "these sessions did not cover. Judging it removable requires Stage 6."
        )

    if unverifiable:
        findings.append(
            _finding(
                "declared_mcp_not_traced",
                strength="observed",
                scope="root",
                summary=f"{len(unverifiable)} MCP server(s) are declared only for a harness whose traces "
                "were not analyzed, so their use cannot be assessed.",
                evidence={"servers": unverifiable, "harnesses_analyzed": sorted(harnesses)},
                judgment="not_waste",
            )
        )
    return findings


def analyze_session(path: str, harness: str | None = None) -> dict:
    """Analyze one transcript by normalizing it first, then reading only its events."""
    normalized = normalize_session(path, harness)
    started = time.perf_counter()
    findings, measurements = _analyze_events(normalized["events"], scope="session")
    return {
        "schema_version": ANALYZER_SCHEMA_VERSION,
        "mode": "analyze",
        "harness": normalized["harness"],
        "source": normalized["source"],
        "read_only": True,
        "session": normalized["session"],
        "observed": measurements,
        "findings": findings,
        "profiler_cost": {
            "stage": "analyze",
            "events_analyzed": len(normalized["events"]),
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
            "normalize_ms": normalized["profiler_cost"]["elapsed_ms"],
            "llm_input_tokens": 0,
            "llm_output_tokens": 0,
        },
        "notes": _NOTES,
    }


def analyze_root(root: str, limit: int | None = None, harness: str | None = None,
                 *, whole_environment: bool = False) -> dict:
    """Analyze every traced session, plus root-level Installed -> Used findings."""
    if not root or not root.strip():
        raise ValueError("root must not be empty")
    base = Path(root).expanduser().resolve()
    if not base.is_dir():
        raise NotADirectoryError(f"root is not a directory: {base}")

    started = time.perf_counter()
    normalized = normalize_root(str(base), limit, harness,
                                whole_environment=whole_environment)

    sessions = []
    findings: list[dict] = []
    used_tools: Counter = Counter()
    harnesses: set[str] = set()
    verdicts: Counter = Counter()
    totals: Counter = Counter()

    for entry in normalized["sessions"]:
        harnesses.add(entry["harness"])
        for tool, count in entry["observed"]["tool_calls"].items():
            used_tools[tool] += count
        # Re-normalize to get events: the root summary deliberately omits them.
        single = normalize_session(entry["source"], entry["harness"])
        session_findings, measurements = _analyze_events(single["events"], scope="session")
        for key, value in measurements["tokens"]["tokens"].items():
            totals[key] += value
        verdicts.update(measurements["repeats"]["verdicts"])
        sessions.append(
            {
                "harness": entry["harness"],
                "source": entry["source"],
                "session": entry["session"],
                "findings": session_findings,
                "observed": measurements,
            }
        )
        findings.extend({**f, "source": entry["source"]} for f in session_findings)

    scan = None
    try:
        from .core import inspect as inspect_stack

        scan = inspect_stack(str(base), whole_environment=whole_environment)
    except (OSError, ValueError):
        scan = None
    if scan is not None:
        findings.extend(_usage_findings(scan, used_tools, harnesses, len(sessions)))

    by_kind: Counter = Counter(f["kind"] for f in findings)
    by_strength: Counter = Counter(f["strength"] for f in findings)
    return {
        "schema_version": ANALYZER_SCHEMA_VERSION,
        "mode": "analyze_root",
        "root": str(base),
        "scope": "whole_environment" if whole_environment else "project",
        "read_only": True,
        "observed": {
            "sessions_analyzed": len(sessions),
            "harnesses": sorted(harnesses),
            "tokens": dict(sorted(totals.items())),
            "repeat_verdicts": dict(sorted(verdicts.items())),
            "findings_by_kind": dict(by_kind.most_common()),
            "findings_by_strength": dict(sorted(by_strength.items())),
        },
        "findings": findings,
        "sessions": sessions,
        "profiler_cost": {
            "stage": "analyze_root",
            "sessions": len(sessions),
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
            "llm_input_tokens": 0,
            "llm_output_tokens": 0,
        },
        "notes": _NOTES,
    }


_NOTES = [
    "The analyzer reads normalized events only; it never parses a transcript.",
    "Every finding carries a strength: `observed` means the events prove it, `estimated` "
    "means a stated assumption was applied and the finding lists it.",
    "No finding here is counterfactual: a finding reports what the events show, never what "
    "a different configuration would have cost. Establishing that requires a real A/B (Stage 6).",
    "A repeated call separated by an opaque command, a write to the same target or a "
    "compaction is reported as inconclusive rather than counted as waste.",
    "Declared-but-never-called is only reported for a harness whose traces were analyzed.",
    "0 LLM tokens are spent; see profiler_cost.",
]
