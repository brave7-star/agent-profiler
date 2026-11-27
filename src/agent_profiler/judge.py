"""Semantic judge.

Stage 5 answers only the questions the deterministic layer provably cannot, and it is
built to spend as little as possible doing so.

The pipeline is fixed:

    events -> deterministic filter -> suspicious subset -> dedup -> compress -> Judge

Never `raw trace -> LLM`. Stage 4 established that 83 of 90 repeated calls in the real
corpus are `inconclusive` because an opaque command sits between the repeats; code cannot
tell whether that command changed the file, but a reader of the command text can. That is
the one question worth tokens here.

## Why everything is batched into a single call

Measured on this machine: one `claude -p` invocation carries 10k-34k tokens of harness
system prompt and tool schema *before* any payload, which dominates a small call's cost
state. Attempts to trim it with `--system-prompt`/`--setting-sources` made it worse.

So per-candidate calls are disqualified: judging 83 candidates one at a time would cost
roughly 100x a single batched call for the same evidence. All candidates go in one request,
and the fixed overhead is measured and reported as `invocation_overhead` rather than hidden.

A profiler that burns 500k tokens to find 5k of waste is a failed product, so the budget is
checked *before* spending and every run reports its own token and dollar cost.

## Evidence discipline

A judge verdict is always `estimated`, never `observed`. It carries the model, the exact
evidence the model was shown, and its own stated confidence. Verdicts are cached by
candidate fingerprint, so re-running costs nothing and an unchanged candidate is never
re-judged.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import time
from collections import Counter
from pathlib import Path

from .analyze import OPAQUE_TOOLS, _classify_repeats
from .events import normalize_root, normalize_session

JUDGE_SCHEMA_VERSION = "0.5"

# Budget tiers. "Deep" and above must be asked for explicitly; nothing here runs A/B.
# Economics, so the judge can refuse a bad trade before making it. Every figure here was
# measured by this project rather than assumed:
#   - median input growth between consecutive responses in a real session.
#     It includes commands, outputs and reasoning, so attributing all of it to one avoided
#     re-read is an upper bound on a candidate's worth.
#   - a batched `claude -p` call, whose harness overhead dominates the payload.
# Rates are published Opus prices, which are an input rather than a measurement.
TOKENS_PER_CANDIDATE = 1609
INVOCATION_COST_USD = 0.30
RATE_INPUT = 15.00
RATE_CACHE_READ = 1.50
# The same threshold Stage 6 uses, set from Stage 5's own failure at roughly 0.5x.
MIN_EIG_RATIO = 5.0

# Below this many candidates a batched call cannot clear the threshold, because the fixed
# harness overhead dominates. Derived rather than chosen: the first tier cap was set to 60
# by hand and the gate then refused it at 4.83x, which is the kind of number a guess
# produces. Tiers start at the break-even point instead.
MIN_CANDIDATES_WORTH_ASKING = int(
    -(-MIN_EIG_RATIO * INVOCATION_COST_USD * 1e6 // (TOKENS_PER_CANDIDATE * RATE_INPUT))
)

BUDGET_TIERS = {
    "quick": {"max_candidates": 0, "max_calls": 0},
    "standard": {"max_candidates": max(60, MIN_CANDIDATES_WORTH_ASKING), "max_calls": 1},
    "deep": {"max_candidates": 200, "max_calls": 3},
}
DEFAULT_TIER = "standard"

# A repeat separated by many commands is later work, not a duplicate worth asking about.
# Derived from the real distribution of intervening commands across 65 raw candidates:
# median 3, p75 6, then a jump to p90 32 and a max of 258. Below the jump, 52 of 53
# candidates are file re-reads; above it they are distant status updates.
MAX_INTERVENING_OPAQUE = 10

# Hard caps on what any single candidate contributes to the prompt. The command cap equals
# the candidate filter above on purpose: showing only some of the intervening commands lets
# the judge conclude "nothing regenerated the target" while the regenerating command sits in
# the part it was never shown. That produced 2 unsound verdicts in the first real run.
MAX_COMMANDS_PER_CANDIDATE = MAX_INTERVENING_OPAQUE
MAX_COMMAND_CHARS = 180
MAX_PROMPT_CHARS = 60_000

# Credential-shaped content is redacted before anything is sent anywhere.
REDACTIONS = (
    (re.compile(r"(?i)(authorization:\s*)(bearer|basic)\s+\S+"), r"\1\2 [REDACTED]"),
    (re.compile(r"(?i)\b(sk-ant-[A-Za-z0-9_-]{8,}|gh[pousr]_[A-Za-z0-9]{8,})"), "[REDACTED]"),
    # Prefixed tokens in general: `<prefix>_<high-entropy body>`. Case-sensitive on purpose —
    # requiring both an upper-case letter and a digit in the body is what separates a
    # credential from a long snake_case identifier, and `(?i)` would defeat that test.
    # Enumerating individual vendor prefixes instead ages badly: rotating a key can change
    # its prefix, and a hard-coded list then stops covering the one actually in use.
    (re.compile(r"\b[a-z][a-z0-9]{1,7}_"
                r"(?=[A-Za-z0-9_-]{20,})(?=[A-Za-z0-9_-]*[A-Z])(?=[A-Za-z0-9_-]*[0-9])"
                r"[A-Za-z0-9_-]{20,}"), "[REDACTED]"),
    (re.compile(r"(?i)\b([A-Z0-9_]*(?:TOKEN|SECRET|PASSWORD|APIKEY|API_KEY)[A-Z0-9_]*\s*=\s*)\S+"), r"\1[REDACTED]"),
    (re.compile(r"(?i)(-H\s+['\"]?authorization:\s*)[^'\"]+"), r"\1[REDACTED]"),
    (re.compile(r"(?i)(--password[= ])\S+"), r"\1[REDACTED]"),
)

SYSTEM_RULES = """You are auditing an AI coding agent's own tool-call trace.

For each candidate you are given: a tool that was called twice with identical arguments,
and the commands that ran between the two calls. Decide whether the SECOND call was
necessary.

Answer only from the evidence shown. The commands are the only thing that could have
changed the result; if they plainly do not touch the target, the repeat was unnecessary.
If you cannot tell from what is shown, say so - "unknown" is a correct and useful answer.

Reply with ONLY a JSON array, one object per candidate id, no prose:
[{"id":"<id>","verdict":"necessary|unnecessary|unknown","confidence":"high|medium|low","why":"<=15 words"}]"""


def _redact(text: str) -> tuple[str, bool]:
    redacted = text
    for pattern, replacement in REDACTIONS:
        redacted = pattern.sub(replacement, redacted)
    return redacted, redacted != text


def _fingerprint(candidate: dict) -> str:
    """Stable id for a candidate, so a cached verdict can be reused safely."""
    payload = json.dumps(
        {"tool": candidate["tool"], "target": candidate["target"], "between": candidate["between_commands"]},
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8", errors="replace")).hexdigest()[:12]


def _raw_command(source: str, line: int) -> str | None:
    """Read one raw record to recover the command text for a specific event.

    The event model deliberately digests tool inputs, because an argument can hold a
    credential. The judge genuinely needs the command text to answer its question, so it
    is recovered here for the few suspicious events only, then redacted and truncated.
    """
    try:
        with open(source, "r", encoding="utf-8", errors="replace") as fh:
            for current, text in enumerate(fh, 1):
                if current != line:
                    continue
                record = json.loads(text)
                message = record.get("message") if isinstance(record, dict) else None
                blocks = message.get("content") if isinstance(message, dict) else None
                for block in blocks if isinstance(blocks, list) else []:
                    if isinstance(block, dict) and block.get("type") == "tool_use":
                        value = block.get("input")
                        if isinstance(value, dict):
                            for key in ("command", "cmd", "script"):
                                if isinstance(value.get(key), str):
                                    return value[key]
                            return json.dumps(value, ensure_ascii=False)[:MAX_COMMAND_CHARS]
                return None
    except (OSError, json.JSONDecodeError, ValueError):
        return None
    return None


def _candidates_from_session(source: str, events: list[dict]) -> list[dict]:
    """Deterministic filter: only `inconclusive` repeats blocked by an opaque command."""
    calls = [e for e in events if e["type"] == "tool_call"]
    classified = _classify_repeats(events)

    candidates = []
    for group in classified["groups"]:
        for repeat in group["repeats"]:
            if repeat["verdict"] != "inconclusive":
                continue
            # Only the opaque-command case is a semantic question. A write to the same
            # target or a compaction is already a sufficient explanation on its own.
            if repeat["reasons"] != ["opaque_command_between"]:
                continue
            first, second = repeat["seqs"]
            between = [
                c for c in calls
                if first < c["seq"] < second and c.get("tool") in OPAQUE_TOOLS
            ]
            # Two cheap rejections before any token is spent: a repeat far from its
            # original is not a duplicate, and without a target there is nothing whose
            # freshness the commands could be read against.
            if len(between) > MAX_INTERVENING_OPAQUE or not group["target"]:
                continue
            commands = []
            redacted_any = False
            for call in between[:MAX_COMMANDS_PER_CANDIDATE]:
                raw = call.get("raw") or {}
                text = _raw_command(raw.get("source", source), raw.get("line", 0))
                if not text:
                    commands.append(f"{call.get('tool')}: <unavailable>")
                    continue
                clean, was_redacted = _redact(text)
                redacted_any = redacted_any or was_redacted
                commands.append(f"{call.get('tool')}: {clean[:MAX_COMMAND_CHARS]}")
            candidate = {
                "tool": group["tool"],
                "target": group["target"],
                "seqs": repeat["seqs"],
                "commands_between": len(between),
                "between_commands": commands,
                # True when the judge is shown every command that ran between the calls.
                # A verdict on partial evidence cannot rule out an unseen cause.
                "evidence_complete": len(commands) == len(between)
                and not any(c.endswith("<unavailable>") for c in commands),
                "redacted": redacted_any,
                "source": source,
            }
            candidate["id"] = _fingerprint(candidate)
            candidates.append(candidate)
    return candidates


def _dedupe(candidates: list[dict]) -> tuple[list[dict], dict]:
    """Collapse candidates that present identical evidence; judging twice buys nothing."""
    unique: dict[str, dict] = {}
    occurrences: Counter = Counter()
    for candidate in candidates:
        occurrences[candidate["id"]] += 1
        unique.setdefault(candidate["id"], candidate)
    for candidate in unique.values():
        candidate["occurrences"] = occurrences[candidate["id"]]
    return list(unique.values()), {"input": len(candidates), "unique": len(unique)}


def _build_prompt(candidates: list[dict]) -> str:
    lines = []
    for candidate in candidates:
        target = candidate["target"] or "(no path argument)"
        lines.append(
            f'id={candidate["id"]} tool={candidate["tool"]} target={target}\n'
            + "\n".join(f"  - {c}" for c in candidate["between_commands"])
        )
    body = "\n\n".join(lines)
    if len(body) > MAX_PROMPT_CHARS:
        body = body[:MAX_PROMPT_CHARS] + "\n[truncated]"
    return (
        f"{len(candidates)} candidates. For each, was the second identical call necessary?\n\n"
        f"{body}\n\nJSON array only."
    )


def _invoke(prompt: str, timeout: int) -> dict:
    """One batched judge call. Returns verdicts plus the exact cost it incurred."""
    binary = shutil.which("claude")
    if binary is None:
        raise RuntimeError("the `claude` CLI is required for the semantic judge but was not found")
    started = time.perf_counter()
    try:
        done = subprocess.run(
            [binary, "-p", prompt, "--append-system-prompt", SYSTEM_RULES, "--output-format", "json"],
            capture_output=True,
            text=True,
            timeout=timeout,
            # See the note in evaluate._probe: `capture_output` leaves stdin inherited, and
            # `claude -p` appends piped stdin to the prompt. Here that would silently
            # corrupt the evidence sent to the judge, which is worse than costing money.
            stdin=subprocess.DEVNULL,
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "timeout", "elapsed_ms": round((time.perf_counter() - started) * 1000, 1)}
    from .claude_result import read_result

    envelope, failure = read_result(done)
    if failure:
        return failure

    usage = envelope.get("usage") or {}
    text = envelope.get("result") or ""
    verdicts = []
    match = re.search(r"\[.*\]", text, re.S)
    if match:
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, list):
                verdicts = [v for v in parsed if isinstance(v, dict)]
        except json.JSONDecodeError:
            pass
    return {
        "ok": True,
        "verdicts": verdicts,
        "unparsed": not verdicts,
        "model": envelope.get("modelUsage") and sorted(envelope["modelUsage"]) or None,
        "cost": {
            "input_tokens": usage.get("input_tokens", 0),
            "cache_read_input_tokens": usage.get("cache_read_input_tokens", 0),
            "cache_creation_input_tokens": usage.get("cache_creation_input_tokens", 0),
            "output_tokens": usage.get("output_tokens", 0),
            "cost_usd": envelope.get("total_cost_usd"),
            "api_ms": envelope.get("duration_api_ms"),
        },
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
    }


def _cache_path() -> Path:
    override = os.environ.get("AGENT_PROFILER_JUDGE_CACHE")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".cache" / "agent-profiler" / "judge-cache.json"


def _load_cache() -> dict:
    path = _cache_path()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _save_cache(cache: dict) -> None:
    path = _cache_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(cache, ensure_ascii=False, indent=0), encoding="utf-8")
    except OSError:
        pass


def _judge_gate(pending: int, cached: int) -> dict:
    """Decide whether this judge run is worth its own cost, before spending anything.

    Stage 5 measured its economics and failed them: it spent far more than it found. The fix
    is not to delete the judge but to put it behind the gate Stage 6 built, so it runs when
    the evidence it would buy is worth more than the asking.

    Both sides are estimated from figures this project measured rather than guessed:
    a candidate is worth at most ~1.6k input tokens (the measured median growth per turn,
    which *includes* non-repeated content, so it is an upper bound), and an invocation costs
    its fixed harness overhead plus the payload.
    """
    value = pending * TOKENS_PER_CANDIDATE / 1e6 * RATE_CACHE_READ
    value_if_fresh = pending * TOKENS_PER_CANDIDATE / 1e6 * RATE_INPUT
    cost = INVOCATION_COST_USD if pending else 0.0
    ratio = (value_if_fresh / cost) if cost else float("inf")
    approved = pending > 0 and ratio >= MIN_EIG_RATIO
    return {
        "candidates_pending": pending,
        "candidates_cached": cached,
        "value_usd_range": [round(value, 4), round(value_if_fresh, 4)],
        "estimated_cost_usd": round(cost, 4),
        "eig_ratio": None if ratio == float("inf") else round(ratio, 2),
        "threshold": MIN_EIG_RATIO,
        "decision": "approved" if approved else "refused",
        "basis": f"a candidate is worth at most ~{TOKENS_PER_CANDIDATE} input tokens "
                 f"(measured median growth per turn, an upper bound); one batched call costs "
                 f"about ${INVOCATION_COST_USD}",
        "reason": (
            f"{pending} candidates are worth up to ${value_if_fresh:.2f}, "
            f"against ~${cost:.2f} to ask."
            if approved else
            "nothing left to judge; the cache already answers every candidate"
            if not pending else
            f"{pending} candidates are worth at most ${value_if_fresh:.2f}, but asking costs "
            f"~${cost:.2f} — below the {MIN_EIG_RATIO}x threshold, so the call is not made. "
            f"Use --force to override, or judge a larger corpus where the fixed overhead "
            f"amortises."
        ),
    }


def judge_root(
    root: str,
    *,
    tier: str = DEFAULT_TIER,
    limit: int | None = None,
    harness: str | None = None,
    dry_run: bool = False,
    force: bool = False,
    timeout: int = 900,
    whole_environment: bool = False,
) -> dict:
    """Funnel a root's events down to semantic candidates, then judge them in one call."""
    if tier not in BUDGET_TIERS:
        raise ValueError(f"unknown budget tier: {tier} (choose from {sorted(BUDGET_TIERS)})")
    if not root or not root.strip():
        raise ValueError("root must not be empty")
    base = Path(root).expanduser().resolve()
    if not base.is_dir():
        raise NotADirectoryError(f"root is not a directory: {base}")

    started = time.perf_counter()
    budget = BUDGET_TIERS[tier]
    normalized = normalize_root(str(base), limit, harness,
                                whole_environment=whole_environment)

    raw_candidates = []
    events_seen = 0
    for entry in normalized["sessions"]:
        single = normalize_session(entry["source"], entry["harness"])
        events_seen += len(single["events"])
        raw_candidates.extend(_candidates_from_session(entry["source"], single["events"]))

    candidates, dedup = _dedupe(raw_candidates)
    cache = _load_cache()
    cached = [c for c in candidates if c["id"] in cache]
    pending = [c for c in candidates if c["id"] not in cache]

    truncated = 0
    if len(pending) > budget["max_candidates"]:
        truncated = len(pending) - budget["max_candidates"]
        pending = pending[: budget["max_candidates"]]

    gate = _judge_gate(len(pending), len(cached))
    # A refusal is a result, not an error: the funnel still reports, and cached verdicts are
    # still served, so a refused run is informative and free.
    gate_blocks = gate["decision"] == "refused" and not force
    if force and gate["decision"] == "refused":
        gate["decision"] = "overridden"
        gate["reason"] = f"--force was passed, overriding: {gate['reason']}"

    funnel = {
        "events_scanned": events_seen,
        "suspicious_after_filter": dedup["input"],
        "after_dedup": dedup["unique"],
        "served_from_cache": len(cached),
        "sent_to_judge": 0 if (dry_run or gate_blocks or not budget["max_calls"]) else len(pending),
        "dropped_over_budget": truncated,
        "dropped_by_gate": len(pending) if gate_blocks else 0,
    }

    calls = []
    verdicts: dict[str, dict] = {c["id"]: dict(cache[c["id"]], served_from="cache") for c in cached}

    if pending and budget["max_calls"] and not dry_run and not gate_blocks:
        prompt = _build_prompt(pending)
        funnel["prompt_chars"] = len(prompt)
        result = _invoke(prompt, timeout)
        calls.append({k: v for k, v in result.items() if k != "verdicts"})
        if result.get("ok"):
            returned = {str(v.get("id")): v for v in result["verdicts"]}
            for candidate in pending:
                answer = returned.get(candidate["id"])
                if not answer:
                    continue
                record = {
                    "verdict": answer.get("verdict", "unknown"),
                    "confidence": answer.get("confidence", "low"),
                    "why": str(answer.get("why", ""))[:120],
                    "model": result.get("model"),
                }
                verdicts[candidate["id"]] = dict(record, served_from="judge")
                cache[candidate["id"]] = record
            _save_cache(cache)

    findings = []
    by_verdict: Counter = Counter()
    for candidate in candidates:
        answer = verdicts.get(candidate["id"])
        if not answer:
            by_verdict["not_judged"] += 1
            continue
        by_verdict[answer["verdict"]] += 1
        if answer["verdict"] != "unnecessary":
            continue
        findings.append(
            {
                "kind": "repeat_judged_unnecessary",
                # A judge verdict is a reading of evidence, never a measurement.
                "strength": "estimated",
                "scope": "session",
                "summary": f"A repeated {candidate['tool']} call was judged unnecessary: {answer['why']}",
                "evidence": {
                    "tool": candidate["tool"],
                    "target": candidate["target"],
                    "seqs": candidate["seqs"],
                    "commands_shown": candidate["between_commands"],
                    "occurrences": candidate["occurrences"],
                },
                "assumptions": [
                    "The judge saw only the commands listed in `commands_shown`, each truncated to "
                    f"{MAX_COMMAND_CHARS} characters, and no command output.",
                    "A command that modifies the target indirectly could be misread as harmless.",
                ]
                + ([] if candidate["evidence_complete"] else [
                    f"Evidence was incomplete: {candidate['commands_between']} commands ran between "
                    f"the calls but only {len(candidate['between_commands'])} could be shown, so an "
                    "unseen command may have changed the target."
                ]),
                "evidence_complete": candidate["evidence_complete"],
                "judge": {"confidence": answer["confidence"], "model": answer.get("model"),
                          "served_from": answer["served_from"]},
                "judgment": "review_only",
                "source": candidate["source"],
            }
        )

    spend = Counter()
    for call in calls:
        for key, value in (call.get("cost") or {}).items():
            if isinstance(value, (int, float)) and key != "cost_usd":
                spend[key] += value
    cost_usd = sum((c.get("cost") or {}).get("cost_usd") or 0 for c in calls)
    judged = sum(1 for v in verdicts.values() if v["served_from"] == "judge")

    return {
        "schema_version": JUDGE_SCHEMA_VERSION,
        "mode": "judge",
        "root": str(base),
        "scope": "whole_environment" if whole_environment else "project",
        "read_only": True,
        "budget_tier": tier,
        "dry_run": dry_run,
        "funnel": funnel,
        "gate": gate,
        "observed": {
            "candidates": len(candidates),
            "verdicts": dict(by_verdict.most_common()),
            "findings": len(findings),
            "findings_on_partial_evidence": sum(1 for f in findings if not f["evidence_complete"]),
        },
        "findings": findings,
        "candidates": candidates if dry_run else None,
        "calls": calls,
        "profiler_cost": {
            "stage": "judge",
            "llm_calls": len(calls),
            "llm_input_tokens": spend["input_tokens"],
            "llm_cache_read_tokens": spend["cache_read_input_tokens"],
            "llm_cache_creation_tokens": spend["cache_creation_input_tokens"],
            "llm_output_tokens": spend["output_tokens"],
            "cost_usd": round(cost_usd, 6) if calls else 0.0,
            "candidates_judged": judged,
            "cost_usd_per_candidate": round(cost_usd / judged, 6) if judged else None,
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
            "invocation_overhead": "Each `claude -p` call carries 10k-34k tokens of harness system "
            "prompt and tool schema before any payload, which is why all candidates are batched "
            "into one call.",
        },
        "notes": [
            "Pipeline is events -> deterministic filter -> dedup -> compress -> judge; a raw "
            "transcript is never sent to a model.",
            "Only `inconclusive` repeats blocked solely by an opaque command reach the judge, "
            f"and only when at most {MAX_INTERVENING_OPAQUE} commands separate the two calls and the "
            "call names a target. Cases the deterministic layer can already explain, or that are "
            "plainly separate work, are not spent on.",
            "Every judge verdict is `estimated` and names the evidence it saw. 'unknown' is an "
            "accepted answer and is not converted into a finding.",
            "Command text is recovered from raw records only for these few candidates, then "
            "credential-redacted and truncated before being sent.",
            "Verdicts are cached by candidate fingerprint, so re-running costs nothing.",
            "No counterfactual claim is made here; establishing the effect of a configuration "
            "change requires a real A/B (Stage 6).",
        ],
    }
