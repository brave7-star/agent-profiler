"""Normalized trace events.

Stage 3 turns harness-specific transcripts into one stable event model, so that later
stages never learn a raw format. Two adapters exist — Claude Code and Codex — because an
abstraction with a single implementation is an assumption, not an abstraction.

Every event keeps a `raw` reference (file, line, native record type) so any number can be
traced back to the bytes it came from. Nothing here judges a run; events are facts.

## The token model is real normalization, not renaming

The two harnesses mean different things by the same words, verified arithmetically
(docs/dogfood/trace-schema.md, docs/dogfood/codex-schema.md):

- Claude Code bills three *separate* input quantities: `input_tokens` (fresh),
  `cache_read_input_tokens`, `cache_creation_input_tokens`. Their sum is the input side.
- Codex reports one `input_tokens` that *already contains* `cached_input_tokens`;
  `input_tokens + output_tokens == total_tokens` held for every record sampled.

Mapping Codex's `cached_input_tokens` onto Anthropic's `cache_read` and adding it to
`input_tokens` would therefore double-count the cached portion. The normalized form states
the invariant instead: `input_fresh + input_cached + input_cache_write == input_total`.

## Per-harness duplication traps, both handled in the adapters

- Claude Code writes one record per content block, repeating `message.usage`; usage is
  taken once per `message.id`, and `api-request-blob` replays are not activity.

  The repeats are not always identical. A transcript can record a response *as it streams*,
  so the early copies carry a partial `output_tokens` that grows — 1, then 3, then the real
  figure. Taking the first copy, which is what counting-once used to mean, reported the
  partial one. The settled copy is the one whose usage carries the completion fields
  (`iterations`, `output_tokens_details`), so that is the copy used, and a message whose
  copies all lack them falls back to the first. Rare in the corpus this was found in and
  worth a fraction of a percent of output tokens there, but wrong in a direction that gets
  worse the more a harness streams.
- Codex reports cumulative totals alongside per-response ones: `turn_token_usage` exceeded
  the per-response total in almost every record sampled, and `event_msg/token_count` carries a
  session-cumulative `total_token_usage`. Only `token_usage_record.usage` is per-response,
  keyed by `response_id`.

  That guard against double counting became a blind spot. `token_usage_record` exists in
  one narrow version window — 0.162 — while `event_msg/token_count` is written by **every**
  version observed, 0.144 through 0.167. Discarding the cumulative record therefore
  reported *zero tokens* for the large majority of Codex sessions, hiding several times the
  volume that was being counted. It is now read as what it is, a
  harness-reported session total, modelled as `cost_report` like Claude's `cost-state`, and
  never added to the per-response sums. Two eras differ again inside it: 0.144-0.162 carry
  a full breakdown, 0.166+ zero every field but `total_tokens`, so `breakdown_present`
  records which — reporting those zeros as measurements would read as "nothing was cached".
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from .core import FileCache, _encode_project_key, _user_tree
from .traces import (
    INTERRUPTION_PREFIX,
    KNOWN_RECORD_TYPES,
    REPLAY_RECORD_TYPES,
    _component_of,
    _prompt_text,
    _signature_match,
    self_generated_verdict,
    _skill_body_read,
    _skill_body_paths,
    _skill_body_written,
    _mcp_servers_called,
    SKILL_BODY_READ,
    _context_loads,
    _parse_ts,
)

EVENT_SCHEMA_VERSION = "1.0"

# The stable vocabulary. Later stages may rely on these names; raw record names never leak.
EVENT_TYPES = (
    "session_start",
    "user_prompt",
    "model_response",
    "tool_call",
    "tool_result",
    "context_compaction",
    "turn_aborted",
    "cost_report",
    "context_load",
    "config_applied",
)

# Tool-input keys whose value identifies the target and is safe to carry forward. Anything
# else is represented only by a digest, because a tool input can hold a credential.
TARGET_KEYS = ("file_path", "notebook_path", "path", "pattern")

DIGEST_LENGTH = 16

CODEX_SESSION_GLOB = "sessions/*/*/*/rollout-*.jsonl*"


def _digest(value: object) -> str:
    """Stable short digest of a tool input.

    Repetition analysis needs only equality, never content, so a digest gives later stages
    what they need while keeping arguments — which may contain secrets — out of the output.
    """
    try:
        canonical = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        canonical = repr(value)
    return hashlib.sha256(canonical.encode("utf-8", errors="replace")).hexdigest()[:DIGEST_LENGTH]


def _target_of(tool_input: object) -> str | None:
    if not isinstance(tool_input, dict):
        return None
    for key in TARGET_KEYS:
        value = tool_input.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _codex_settings(rtype: str | None, ptype: str | None, payload: dict) -> dict | None:
    """Configuration that was actually in effect, which Codex records and Claude Code does not.

    `turn_context` carries it per turn and `thread_settings_applied` per thread. Neither
    names an instruction file, so these do not close the AGENTS.md gap — but they do carry
    the model and the reasoning effort, which are direct cost drivers the project could not
    see before. Only scalar settings are kept; the nested permission and collaboration
    blocks are configuration detail with no bearing on cost or usage.

    The two live at different levels, which is what this function got wrong: a
    `turn_context` record names itself at the **top** level and its payload has no `type`
    at all, so matching it against `ptype` could never fire. `rtype` sat unused in the
    signature while every record carrying the per-turn model and effort was dropped —
    the one thing the docstring above promises.
    """
    if rtype == "turn_context":
        source = payload
    elif ptype == "thread_settings_applied":
        source = payload.get("thread_settings")
    else:
        return None
    if not isinstance(source, dict):
        return None

    def scalar(*keys):
        for key in keys:
            value = source.get(key)
            if isinstance(value, (str, bool)) and value != "":
                return value
        return None

    sandbox = source.get("sandbox_policy")
    out = {
        "scope": "turn" if rtype == "turn_context" else "thread",
        "model": scalar("model"),
        "effort": scalar("reasoning_effort", "effort"),
        "approval_policy": scalar("approval_policy"),
        "sandbox": sandbox.get("type") if isinstance(sandbox, dict) else scalar("sandbox_policy"),
        "personality": scalar("personality"),
        # Declared but switched off is a real Installed-but-not-active signal. Empty in this
        # corpus, which is itself worth recording rather than assuming.
        "disabled_plugins": [p for p in source.get("disabled_plugin_ids") or []
                             if isinstance(p, str)],
    }
    return out if out["model"] or out["effort"] else None


def _tokens(
    *, input_total: int, input_cached: int, input_cache_write: int, output: int, reasoning: int | None
) -> dict:
    """Build the normalized token block and keep its invariant checkable."""
    fresh = input_total - input_cached - input_cache_write
    return {
        "input_total": input_total,
        "input_fresh": fresh,
        "input_cached": input_cached,
        "input_cache_write": input_cache_write,
        "output": output,
        "reasoning": reasoning,
        "total": input_total + output,
    }


def _prompt_origin(blocks: object) -> dict:
    """Who sent this prompt, as two derived signals and no prompt text.

    The collector recognises the profiler's own sessions from the prompts it keeps in
    memory; the normalizer cannot keep prompts, because its events are writable to disk
    with `--events-out` and user prompt text is not something this tool should start
    copying into artifacts. So the text is read here, reduced to a signature name and a
    flag, and dropped. `produced_by` is either None or the name of a stage of this tool,
    and `interruption` is a property of a harness-written marker — neither is user
    content, so both are safe to carry in the event stream.
    """
    text = _prompt_text(blocks)
    match = _signature_match(text)
    return {
        "produced_by": match[0] if match else None,
        "signature_evidence": match[1] if match else None,
        "interruption": text.startswith(INTERRUPTION_PREFIX),
    }


def _event(seq: int, etype: str, harness: str, ts, source: Path, line: int, record: str, **fields) -> dict:
    event = {
        "seq": seq,
        "ts": ts.isoformat().replace("+00:00", "Z") if isinstance(ts, datetime) else None,
        "type": etype,
        "harness": harness,
    }
    event.update(fields)
    event["raw"] = {"source": str(source), "line": line, "record": record}
    return event


# --------------------------------------------------------------------------------------
# Claude Code adapter
# --------------------------------------------------------------------------------------


# A response's usage is repeated on every record of that response, and while it streams the
# repeats differ: `output_tokens` grows toward the real figure. These fields appear only once
# the response is complete, which is what tells a settled copy from a partial one.
SETTLED_USAGE_FIELDS = ("iterations", "output_tokens_details")


def _usage_is_settled(usage: dict) -> bool:
    return any(field in usage for field in SETTLED_USAGE_FIELDS)


def _claude_events(path: Path) -> Iterator[dict]:
    seq = 0
    emitted_session = False
    settled_messages: set[str] = set()
    # Responses seen only as partial copies so far, held back in case a settled copy follows.
    # Keyed by `message.id`, holding the fields the event would be built from.
    partial_messages: dict[str, dict] = {}
    seen_tool_calls: set[str] = set()
    tool_of_call: dict[str, str] = {}
    request_ts: dict[str, datetime] = {}

    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict):
                continue

            rtype = obj.get("type") or "__missing_type__"
            ts = _parse_ts(obj.get("timestamp"))

            if not emitted_session and isinstance(obj.get("sessionId"), str):
                emitted_session = True
                seq += 1
                yield _event(
                    seq, "session_start", "claude_code", ts, path, line_no, rtype,
                    session_id=obj.get("sessionId"),
                    cwd=obj.get("cwd"),
                    harness_version=obj.get("version"),
                    git_branch=obj.get("gitBranch"),
                )

            if rtype == "api-request":
                if isinstance(obj.get("id"), str) and ts is not None:
                    request_ts[obj["id"]] = ts
                continue
            if rtype in REPLAY_RECORD_TYPES:
                continue
            if rtype == "compact-boundary":
                seq += 1
                yield _event(seq, "context_compaction", "claude_code", ts, path, line_no, rtype)
                continue
            if rtype == "cost-state":
                usage = obj.get("modelUsage") if isinstance(obj.get("modelUsage"), dict) else {}
                seq += 1
                yield _event(
                    seq, "cost_report", "claude_code", ts, path, line_no, rtype,
                    cost_usd=obj.get("totalCostUSD"),
                    api_duration_ms=obj.get("totalAPIDuration"),
                    tool_duration_ms=obj.get("totalToolDuration"),
                    models=sorted(usage),
                    reported_by="harness",
                )
                continue

            if rtype == "attachment":
                for fields in _context_loads(obj):
                    seq += 1
                    yield _event(seq, "context_load", "claude_code", ts, path, line_no, rtype,
                                 **fields)
                continue

            message = obj.get("message")
            blocks = message.get("content") if isinstance(message, dict) else None

            if rtype == "user":
                has_text = False
                for block in blocks if isinstance(blocks, list) else []:
                    if not isinstance(block, dict):
                        continue
                    if block.get("type") == "text":
                        has_text = True
                    elif block.get("type") == "tool_result":
                        call_id = block.get("tool_use_id")
                        seq += 1
                        yield _event(
                            seq, "tool_result", "claude_code", ts, path, line_no, rtype,
                            call_id=call_id if isinstance(call_id, str) else None,
                            tool=tool_of_call.get(call_id),
                            ok=not bool(block.get("is_error")),
                            error=bool(block.get("is_error")),
                        )
                if isinstance(blocks, str) and blocks.strip():
                    has_text = True
                if has_text:
                    seq += 1
                    yield _event(seq, "user_prompt", "claude_code", ts, path, line_no, rtype,
                                 **_prompt_origin(blocks))
                continue

            if rtype != "assistant" or not isinstance(message, dict):
                continue

            msg_id = message.get("id")
            usage = message.get("usage")
            # One response is written as several records; usage belongs to the response.
            if isinstance(usage, dict) and not (isinstance(msg_id, str) and msg_id in settled_messages):
                ref = obj.get("requestRef")
                started = request_ts.get(ref) if isinstance(ref, str) else None
                latency = None
                if started is not None and ts is not None:
                    delta = (ts - started).total_seconds()
                    latency = round(delta, 3) if delta >= 0 else None
                held = {
                    "ts": ts, "line_no": line_no, "rtype": rtype,
                    "response_id": msg_id,
                    "model": message.get("model"),
                    "stop_reason": message.get("stop_reason"),
                    "incomplete": message.get("stop_reason") is None,
                    "latency_seconds": latency,
                    "tokens": _tokens(
                        # Anthropic bills these three separately, so the input side is their sum.
                        input_total=(
                            int(usage.get("input_tokens") or 0)
                            + int(usage.get("cache_read_input_tokens") or 0)
                            + int(usage.get("cache_creation_input_tokens") or 0)
                        ),
                        input_cached=int(usage.get("cache_read_input_tokens") or 0),
                        input_cache_write=int(usage.get("cache_creation_input_tokens") or 0),
                        output=int(usage.get("output_tokens") or 0),
                        reasoning=None,
                    ),
                }
                # The records of one response are contiguous, so reaching a different
                # response settles the matter for the one before it: whatever was held for it
                # is the best figure the transcript has. Flushing here rather than at the end
                # of the file keeps `seq` in the order the records were written.
                for other in [i for i in partial_messages if i != msg_id]:
                    seq += 1
                    yield _model_response(seq, path, partial_messages.pop(other))
                # A settled copy ends the matter and supersedes any partial one held for this
                # response. A partial copy is held rather than emitted, because emitting it
                # would publish a growing figure as the final one; an id-less response cannot
                # be grouped at all, so it is emitted as it stands.
                if _usage_is_settled(usage) or not isinstance(msg_id, str):
                    if isinstance(msg_id, str):
                        settled_messages.add(msg_id)
                        partial_messages.pop(msg_id, None)
                    seq += 1
                    yield _model_response(seq, path, held)
                elif msg_id not in partial_messages:
                    partial_messages[msg_id] = held

            for block in blocks if isinstance(blocks, list) else []:
                if not isinstance(block, dict) or block.get("type") != "tool_use":
                    continue
                call_id = block.get("id")
                if isinstance(call_id, str):
                    if call_id in seen_tool_calls:
                        continue
                    seen_tool_calls.add(call_id)
                name = block.get("name") if isinstance(block.get("name"), str) else None
                if isinstance(call_id, str) and name:
                    tool_of_call[call_id] = name
                seq += 1
                yield _event(
                    seq, "tool_call", "claude_code", ts, path, line_no, rtype,
                    call_id=call_id if isinstance(call_id, str) else None,
                    tool=name,
                    target=_target_of(block.get("input")),
                    component=_component_of(name, block.get("input")),
                    skill_body_read=_skill_body_read(name, block.get("input")),
                    skill_body_paths=_skill_body_paths(name, block.get("input")),
                    skill_body_written=_skill_body_written(name, block.get("input")),
                    mcp_servers_called=_mcp_servers_called(name, block.get("input")),
                    input_digest=_digest(block.get("input")),
                )

    # The last response, if its copies never settled and no later one flushed it. A partial
    # figure is still the only figure the transcript has, so dropping it would lose the
    # response entirely.
    for held in list(partial_messages.values()):
        seq += 1
        yield _model_response(seq, path, held)


def _model_response(seq: int, path: Path, held: dict) -> dict:
    """Build the held response event, numbered at the point it is actually emitted."""
    fields = {k: v for k, v in held.items() if k not in ("ts", "line_no", "rtype")}
    return _event(seq, "model_response", "claude_code", held["ts"], path,
                  held["line_no"], held["rtype"], **fields)


# --------------------------------------------------------------------------------------
# Codex adapter
# --------------------------------------------------------------------------------------


def _codex_lines(path: Path) -> Iterator[str]:
    """Yield transcript lines, decompressing .zst through the zstd binary if needed."""
    if path.suffix != ".zst":
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            yield from fh
        return

    binary = shutil.which("zstd")
    if binary is None:
        raise RuntimeError("zstd is required to read compressed Codex transcripts but was not found")
    proc = subprocess.Popen([binary, "-dc", str(path)], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    try:
        assert proc.stdout is not None
        yield from proc.stdout
    finally:
        if proc.stdout is not None:
            proc.stdout.close()
        proc.wait()


def _codex_events(path: Path) -> Iterator[dict]:
    seq = 0
    emitted_session = False
    seen_responses: set[str] = set()
    tool_of_call: dict[str, str] = {}

    for line_no, line in enumerate(_codex_lines(path), 1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue

        rtype = obj.get("type")
        payload = obj.get("payload") if isinstance(obj.get("payload"), dict) else {}
        ptype = payload.get("type")
        record = f"{rtype}/{ptype}" if ptype else str(rtype)
        ts = _parse_ts(obj.get("timestamp"))

        if rtype == "session_meta":
            if not emitted_session:
                emitted_session = True
                seq += 1
                yield _event(
                    seq, "session_start", "codex", ts, path, line_no, record,
                    session_id=payload.get("session_id"),
                    cwd=payload.get("cwd"),
                    harness_version=payload.get("cli_version"),
                    model_provider=payload.get("model_provider"),
                )
            continue

        if rtype == "compacted":
            seq += 1
            yield _event(seq, "context_compaction", "codex", ts, path, line_no, record)
            continue

        settings = _codex_settings(rtype, ptype, payload)
        if settings is not None:
            seq += 1
            yield _event(seq, "config_applied", "codex", ts, path, line_no, record, **settings)
            continue

        if rtype == "event_msg" and ptype == "token_count":
            # The only token record every Codex version writes, and for most of them the
            # only one at all. Refusing to sum it was right — it is session-cumulative —
            # but discarding it meant reporting zero tokens for every version that has no
            # `token_usage_record`, which is 0.144 through 0.155 and again 0.166 onward.
            # `token_usage_record` exists in one narrow window (0.162), so the guard
            # against double counting had become a blind spot over most of the corpus.
            #
            # Modelled as `cost_report` for the same reason Claude's `cost-state` is: a
            # harness-reported running total, not a per-request measurement. Kept out of
            # the per-response sums entirely; `normalize_session` keeps the last one.
            info = payload.get("info") if isinstance(payload.get("info"), dict) else None
            total = (info or {}).get("total_token_usage")
            if not isinstance(total, dict) or total.get("total_tokens") is None:
                # 0.167 reuses this event type for rate-limit notices: `info` is null and
                # the payload carries `rate_limits` instead. Those are not token reports,
                # and emitting one as an empty snapshot would let a notice arriving last
                # overwrite the session's real total with zero. No session in this corpus
                # happens to end with one, which is luck rather than a reason to rely on
                # ordering.
                continue
            fields = {k: total.get(k) for k in (
                "input_tokens", "cached_input_tokens", "cache_write_input_tokens",
                "output_tokens", "reasoning_output_tokens", "total_tokens")}
            # 0.166 and 0.167 zero every field but `total_tokens`. Reporting those zeros
            # as measurements would read as "nothing was cached", so whether the breakdown
            # is real is recorded rather than inferred by the reader.
            detailed = any(fields[k] for k in fields if k != "total_tokens")
            seq += 1
            yield _event(
                seq, "cost_report", "codex", ts, path, line_no, record,
                cumulative=True, breakdown_present=detailed,
                context_window=info.get("model_context_window"),
                tokens=fields, reported_by="harness",
            )
            continue

        if rtype == "token_usage_record":
            # Only `usage` is per-response. `turn_token_usage` is cumulative and must not
            # be summed; the session-cumulative figure lives in event_msg/token_count.
            usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else None
            response_id = payload.get("response_id")
            if usage is None:
                continue
            if isinstance(response_id, str):
                if response_id in seen_responses:
                    continue
                seen_responses.add(response_id)
            input_total = int(usage.get("input_tokens") or 0)
            output = int(usage.get("output_tokens") or 0)
            # Codex offers no trustworthy session-level total to check against (its
            # cumulative token_count does not reconcile across compactions), so the
            # per-record identity input + output == total is the available validation.
            stated_total = usage.get("total_tokens")
            reported_mismatch = (
                isinstance(stated_total, int) and stated_total != input_total + output
            )
            seq += 1
            yield _event(
                seq, "model_response", "codex", ts, path, line_no, record,
                response_id=response_id,
                model=payload.get("model"),
                stop_reason=None,
                incomplete=False,
                latency_seconds=None,
                tokens=_tokens(
                    # Codex's input_tokens already contains the cached portion.
                    input_total=input_total,
                    input_cached=int(usage.get("cached_input_tokens") or 0),
                    input_cache_write=int(usage.get("cache_write_input_tokens") or 0),
                    output=output,
                    reasoning=int(usage.get("reasoning_output_tokens") or 0),
                ),
                totals_disagree=reported_mismatch,
            )
            continue

        if rtype == "event_msg" and ptype == "turn_aborted":
            seq += 1
            yield _event(seq, "turn_aborted", "codex", ts, path, line_no, record)
            continue

        # Must sit above the `response_item` guard: `world_state` is its own record type,
        # and a check placed below the guard is unreachable.
        if rtype == "world_state":
            load = _host_skills_load(payload)
            if load is not None:
                seq += 1
                yield _event(seq, "context_load", "codex", ts, path, line_no, record,
                             **load)
            continue

        if rtype != "response_item":
            continue

        if ptype == "message" and payload.get("role") == "user":
            seq += 1
            # Run the same check as the Claude path even though no probe writes a Codex
            # session today: `claude -p` is the only self-probe route that exists, and
            # asserting that here by omission would make a future Codex probe invisible.
            yield _event(seq, "user_prompt", "codex", ts, path, line_no, record,
                         **_prompt_origin(payload.get("content")))
            continue

        if ptype in {"custom_tool_call", "function_call"}:
            call_id = payload.get("call_id")
            name = payload.get("name") if isinstance(payload.get("name"), str) else None
            if isinstance(call_id, str) and name:
                tool_of_call[call_id] = name
            # custom_tool_call carries a free-text `input`; function_call carries JSON
            # `arguments`. Either can hold a secret, so only a digest is emitted.
            raw_input = payload.get("input") if ptype == "custom_tool_call" else payload.get("arguments")
            parsed = raw_input
            if isinstance(raw_input, str) and ptype == "function_call":
                try:
                    parsed = json.loads(raw_input)
                except json.JSONDecodeError:
                    parsed = raw_input
            seq += 1
            yield _event(
                seq, "tool_call", "codex", ts, path, line_no, record,
                call_id=call_id if isinstance(call_id, str) else None,
                tool=name,
                target=_target_of(parsed),
                component=_component_of(name, parsed),
                skill_body_read=_skill_body_read(name, parsed),
                skill_body_paths=_skill_body_paths(name, parsed),
                skill_body_written=_skill_body_written(name, parsed),
                mcp_servers_called=_mcp_servers_called(name, parsed),
                input_digest=_digest(raw_input),
            )
            continue

        if ptype in {"custom_tool_call_output", "function_call_output"}:
            call_id = payload.get("call_id")
            failed = _codex_output_failed(payload)
            seq += 1
            yield _event(
                seq, "tool_result", "codex", ts, path, line_no, record,
                call_id=call_id if isinstance(call_id, str) else None,
                tool=tool_of_call.get(call_id),
                ok=not failed,
                error=failed,
            )


def _host_skills_load(payload: dict) -> dict | None:
    """The skill listing Codex says it sent, with the token cost it measured itself.

    Found in `world_state`, which is easy to write off as "a workspace snapshot,
    deliberately unused" — and which turns out to carry two things nothing else does:

        host_skills.usage   -> {"tokens": 2100, "characters": 7200}
        host_skills.entries -> every skill's path with a `shown` flag
        host_skills.roots   -> the directories skills are loaded from

    The `usage` figure is a **recorded** cost for the skill listing, free and exact, where
    the project otherwise has to pay for a paired probe to estimate one. `roots` is what
    exposed a skills directory the scanner was not looking at.

    `shown` is read but deliberately not treated as a `used` signal: it was `true` for all
    21 entries in every record sampled, so it does not vary and therefore carries no
    information about which skill mattered. Reporting it as usage would dress a constant up
    as a measurement.
    """
    host = payload.get("state")
    host = (host or {}).get("host_skills") if isinstance(host, dict) else None
    if not isinstance(host, dict):
        return None
    entries = host.get("entries")
    if not isinstance(entries, dict) or not entries:
        return None
    usage = host.get("usage") if isinstance(host.get("usage"), dict) else {}
    names = []
    paths = []
    for key in entries:
        text = key if isinstance(key, str) else ""
        match = SKILL_BODY_READ.search(text)
        if match:
            names.append(match.group(1))
        # The key is a JSON array whose last element is the file. Keeping the path, not
        # just the name, is what lets a listing prove `loaded` for a specific component:
        # `demo-skill` exists under two roots here, and a name cannot say which was listed.
        paths.extend(_skill_body_paths("exec", text))
    return {
        "load_kind": "skill_listing",
        "components": sorted(set(names)),
        "component_paths": sorted(set(paths)),
        "component_kind": "skills",
        "shared": True,
        "depth": "description_only" if not host.get("omitDescriptions") else "name_only",
        "instructions_sent": bool(host.get("instructionsSent")),
        "tokens": usage.get("tokens") if isinstance(usage.get("tokens"), int) else None,
        "chars": usage.get("characters") if isinstance(usage.get("characters"), int) else None,
        "roots": sorted({v for v in (host.get("roots") or {}).values()
                         if isinstance(v, str)}),
    }


def _codex_extract(events: list[dict]) -> dict:
    """Everything both rollups need from one session, and nothing that depends on the
    machine's inventory.

    This separation is what makes a rollout's contribution cacheable. `_ComponentUsageRollup`
    answers "how often was *this installed skill* read", which depends on what is installed
    — so the paths a session read are kept raw here and matched against the inventory at
    merge time. Cache the filtered answer instead and installing a skill would not change
    the result, which is the kind of staleness that looks like a measurement.

    A read is recorded per path rather than per event because the two are equivalent for
    every consumer: a path maps to exactly one component key, so no event can contribute to
    one key through two different paths, and `skill_body_paths` is already deduplicated
    within an event.
    """
    listings = []
    mcp_calls: Counter = Counter()
    skill_paths: Counter = Counter()
    written: set[str] = set()
    # One model response is one request, which is the denominator an always-on prefix cost
    # has to be multiplied by to become a figure about this environment rather than a rate.
    responses = 0
    # Where the session ran. A project-scoped AGENTS.md is carried only by the requests made
    # inside that project, so the requests have to be attributable to a directory before a
    # project-scoped rate can be multiplied by anything honest.
    cwd = None
    for event in events:
        etype = event.get("type")
        if etype == "session_start" and cwd is None:
            value = event.get("cwd")
            cwd = value if isinstance(value, str) else None
            continue
        if etype == "model_response":
            responses += 1
            continue
        if etype == "context_load":
            if event.get("load_kind") != "skill_listing":
                continue
            listings.append({
                "tokens": event.get("tokens") if isinstance(event.get("tokens"), int) else None,
                "chars": event.get("chars") if isinstance(event.get("chars"), int) else None,
                "components": sorted(event.get("components") or []),
                "component_paths": sorted(event.get("component_paths") or []),
                "roots": sorted(event.get("roots") or []),
            })
            continue
        if etype != "tool_call":
            continue
        for server in event.get("mcp_servers_called") or ():
            mcp_calls[server] += 1
        paths = event.get("skill_body_paths") or ()
        if not paths:
            continue
        edited = bool(event.get("skill_body_written"))
        for full in paths:
            skill_paths[full] += 1
            if edited:
                written.add(full)
    return {
        "listings": listings,
        "mcp_calls": dict(mcp_calls),
        "skill_paths": dict(skill_paths),
        "skill_written": sorted(written),
        "responses": responses,
        "cwd": cwd,
    }


class _SharedContextRollup:
    """What `codex_shared_context` accumulates, fed one session's extract at a time."""

    def __init__(self) -> None:
        self.tokens: list[int] = []
        self.chars: list[int] = []
        self.skills: set[str] = set()
        self.roots: set[str] = set()
        self.listed: Counter = Counter()
        self.carrying = 0
        self.responses = 0
        self.responses_by_cwd: Counter = Counter()

    def feed(self, extract: dict) -> None:
        here: set[str] = set()
        listings = extract["listings"]
        for listing in listings:
            here.update(listing["component_paths"])
            if isinstance(listing["tokens"], int):
                self.tokens.append(listing["tokens"])
            if isinstance(listing["chars"], int):
                self.chars.append(listing["chars"])
            self.skills.update(listing["components"])
            self.roots.update(listing["roots"])
        for one in here:
            self.listed[one] += 1
        self.carrying += 1 if listings else 0
        self.responses += extract.get("responses") or 0
        cwd = extract.get("cwd")
        if isinstance(cwd, str):
            self.responses_by_cwd[cwd] += extract.get("responses") or 0

    def result(self, discovered: int, read: int) -> dict:
        return {
            "harness": "codex",
            # Every rollout discovered, not only those that parsed: this figure is the
            # denominator for `sessions_carrying_record` below.
            "sessions_scanned": discovered,
            # Only recent sessions carry `world_state`, so a figure from it describes those
            # sessions rather than the whole history. Saying how many is the difference
            # between a measurement and a claim about the corpus.
            "sessions_carrying_record": self.carrying,
            "skill_listing_tokens": sorted(set(self.tokens)),
            "skill_listing_chars": sorted(set(self.chars)),
            "skills_covered": sorted(self.skills),
            # Per-path, because the listing is the only record that names a Codex skill
            # entering context. The shared *cost* still cannot be split between them; which
            # ones were listed is a separate fact and is attributable.
            "sessions_listing_path": dict(self.listed.most_common()),
            # The request count for this harness. An always-on prefix is carried by every
            # request, so a per-request rate only becomes a figure about this environment
            # once multiplied by this.
            "requests": self.responses,
            # The same count split by the directory each session ran in, so a project-scoped
            # rate is multiplied by its own project's traffic rather than the environment's.
            "requests_by_cwd": dict(self.responses_by_cwd.most_common()),
            "roots": sorted(self.roots),
            "shared": True,
            "attributable": False,
        }


class _ComponentUsageRollup:
    """What `codex_component_usage` accumulates, fed one session's events at a time."""

    def __init__(self, paths: dict[str, str]) -> None:
        # Normalized, never resolved. `resolve()` follows the filesystem — on this platform
        # it rewrites a path under a firmlink to `/System/Volumes/Data/...` — while the
        # paths in a transcript are whatever text the model typed. Resolving one side and
        # not the other makes them never match, which is only invisible when both happen to
        # exist already.
        self.by_path: dict[str, list[str]] = {}
        for key, p in paths.items():
            self.by_path.setdefault(str(Path(p)), []).append(key)
        self.reads: Counter = Counter()
        self.seen_in: Counter = Counter()
        self.authoring: Counter = Counter()
        self.mcp_calls: Counter = Counter()
        self.mcp_sessions: Counter = Counter()

    def feed(self, extract: dict) -> None:
        for server, calls in extract["mcp_calls"].items():
            self.mcp_calls[server] += calls
            # Once per server per session: the extract's keys are the servers this session
            # called, which is what the per-event version accumulated into a set.
            self.mcp_sessions[server] += 1
        here: Counter = Counter()
        wrote: set[str] = set()
        written = set(extract["skill_written"])
        for full, count in extract["skill_paths"].items():
            for key in self.by_path.get(full, ()):
                here[key] += count
                if full in written:
                    wrote.add(key)
        for key, count in here.items():
            if key in wrote:
                self.authoring[key] += 1
                continue
            self.reads[key] += count
            self.seen_in[key] += 1

    def result(self, discovered: int, read: int) -> dict:
        return {
            "harness": "codex",
            # Only rollouts that actually parsed: this is the basis of the read counts.
            "sessions_scanned": read,
            "reads": dict(self.reads.most_common()),
            "sessions": dict(self.seen_in.most_common()),
            "authoring_sessions": dict(self.authoring.most_common()),
            # Keyed by the underscore tool-namespace form, which is how a server is
            # addressed rather than how it is declared; the caller normalizes its declared
            # name to match.
            "mcp_calls": dict(self.mcp_calls.most_common()),
            "mcp_sessions": dict(self.mcp_sessions.most_common()),
            "basis": "a shell command read the skill's installed SKILL.md in a session that "
                     "never wrote to it",
            "caveat": "reading a skill's body is also what authoring it looks like, so only "
                      "reads of the installed path count, sessions that also wrote to the file "
                      "are set aside as authoring, and the session spread is reported so one "
                      "session cannot look like repeated use",
        }


def _walk_codex(root: Path | None, limit: int | None, rollups: list, *,
                use_cache: bool = True) -> tuple[int, int, int]:
    """Get each rollout's extract once, feeding every rollup that wants it.

    The two rollups ask different questions of the same events, and `report` used to call
    them one after the other — parsing the whole rollout corpus twice, tens of seconds a
    pass, beneath a docstring in `report._collect` promising that nothing is collected
    twice. Sharing the parsed events by holding them is not an option: a whole corpus at
    once runs to hundreds of megabytes. So
    they are fed from one walk, and that walk reuses an extract whenever the rollout's bytes
    are unchanged — keyed exactly as `normalize_root` keys its summaries, for the same
    reason.
    """
    sessions = discover_codex_sessions(root)
    if limit:
        sessions = sessions[-limit:]
    cache = FileCache("codex-extract", enabled=use_cache)
    read = 0
    for entry in sessions:
        path = Path(entry["path"] if isinstance(entry, dict) else entry)
        try:
            extract = cache.get_or_compute(
                path, lambda p=path: _codex_extract(list(_codex_events(p))))
        except (OSError, ValueError):
            continue
        read += 1
        for rollup in rollups:
            rollup.feed(extract)
    cache.flush()
    return len(sessions), read, cache.reused


def codex_rollups(paths: dict[str, str] | None, root: Path | None = None, *,
                  limit: int | None = None,
                  use_cache: bool = True) -> tuple[dict, dict | None]:
    """Both Codex rollups from a single pass over the rollouts.

    Returns `(shared_context, component_usage)`, with the second None when the environment
    declares no Codex skills to look for — the same contract `codex_skill_usage` has, so a
    caller with nothing to attribute does not pay for attribution.
    """
    shared = _SharedContextRollup()
    # `paths is not None`, not `if paths`: an empty mapping is a caller that wants
    # attribution and found nothing to attribute, which is a different answer from a
    # caller that does not want it at all. Collapsing the two would report
    # `sessions_scanned: 0` for a corpus that was in fact fully read.
    usage = _ComponentUsageRollup(paths) if paths is not None else None
    discovered, read, reused = _walk_codex(
        root, limit, [r for r in (shared, usage) if r is not None], use_cache=use_cache)
    reuse = {"enabled": use_cache, "sessions_reused": reused,
             "sessions_parsed": read - reused,
             "keyed_on": "transcript bytes, package source and path"}
    shared_result = shared.result(discovered, read)
    shared_result["reuse"] = reuse
    usage_result = usage.result(discovered, read) if usage is not None else None
    if usage_result is not None:
        usage_result["reuse"] = reuse
    return shared_result, usage_result


def codex_shared_context(root: Path | None = None, *, limit: int | None = None,
                         use_cache: bool = True) -> dict:
    """The skill listing cost Codex recorded for itself, read free from rollouts.

    This is the only place in either harness where the cost of a shared context block is
    *measured by the harness and written down*. Everywhere else the project either pays for
    a paired probe or reports characters and says plainly that it did not tokenize.

    The figure is deliberately returned as an aggregate with the skills it covers, never
    divided among them: one block names ~20 skills, and splitting it would invent the
    attribution this project refuses to invent.
    """
    shared, _ = codex_rollups(None, root, limit=limit, use_cache=use_cache)
    return shared


def codex_component_usage(paths: dict[str, str], root: Path | None = None, *,
                          limit: int | None = None, use_cache: bool = True) -> dict:
    """How often Codex read each declared skill's body, which is how Codex uses a skill.

    Codex never emits a tool call naming a skill: it `cat`s `SKILL.md` through its shell.
    So the evidence is a read, and a read is weaker than an invocation in one specific way
    — the same read happens while *authoring* the skill. The project has a recorded case of
    544 apparent hits that were development work.

    The discriminator is the path. `paths` maps a component key to the file the inventory
    actually found on disk, and only a read of that exact path counts. On a real corpus that
    rule removes most candidate reads: the bulk of them are harness-bundled plugin caches,
    which are not the user's components at all, or stray copies outside any skills root.
    Matching installed paths is also what keeps a skill named `pdf` in a
    vendor bundle from being credited to a user skill that happens to share a name.

    Authoring an *installed* skill still reads the installed path, so a second rule
    separates those: a session that also writes to the file is an editing session, and its
    reads are counted apart under `authoring_sessions`. On a real corpus this is decisive
    rather than cosmetic: some skills are only ever touched while being written, and
    counting those as use would invent a usage record the harness never made.

    `sessions` is reported beside `reads` for the same reason: one session doing forty
    reads is a different claim from forty sessions doing one each.
    """
    _, usage = codex_rollups(paths, root, limit=limit, use_cache=use_cache)
    return usage


def _codex_output_failed(payload: dict) -> bool:
    """Codex has no is_error flag; a non-zero exit_code in the output is the signal."""
    output = payload.get("output")
    chunks = output if isinstance(output, list) else [output]
    for chunk in chunks:
        text = chunk.get("text") if isinstance(chunk, dict) else chunk
        if not isinstance(text, str) or '"exit_code"' not in text:
            continue
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            continue
        code = parsed.get("exit_code") if isinstance(parsed, dict) else None
        if isinstance(code, int) and code != 0:
            return True
    return False


ADAPTERS = {"claude_code": _claude_events, "codex": _codex_events}


def detect_harness(path: Path) -> str:
    """Identify the harness from the transcript's own shape, not from its location."""
    name = path.name
    if name.startswith("rollout-") or path.suffix == ".zst":
        return "codex"
    try:
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if not isinstance(obj, dict):
                    break
                if obj.get("type") in {"session_meta", "token_usage_record", "response_item", "event_msg"}:
                    return "codex"
                if obj.get("type") in KNOWN_RECORD_TYPES or "sessionId" in obj:
                    return "claude_code"
                break
    except (OSError, json.JSONDecodeError):
        pass
    return "claude_code"


def normalize_session(path: str, harness: str | None = None) -> dict:
    """Normalize one transcript into events plus a deterministic summary."""
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"trace file does not exist: {p}")
    if not p.is_file():
        raise IsADirectoryError(f"trace path is not a file: {p}")

    chosen = harness or detect_harness(p)
    if chosen not in ADAPTERS:
        raise ValueError(f"unknown harness: {chosen}")

    started = time.perf_counter()
    counts: Counter = Counter({t: 0 for t in EVENT_TYPES})
    tools: Counter = Counter()
    tool_errors: Counter = Counter()
    repeats: Counter = Counter()
    # Keys are seeded so every summary has the same shape, empty session or not.
    tokens: Counter = Counter(
        {k: 0 for k in ("input_total", "input_fresh", "input_cached", "input_cache_write", "output", "reasoning")}
    )
    latencies: list[float] = []
    models: Counter = Counter()
    settings: Counter = Counter()
    disabled_plugins: Counter = Counter()
    warnings: list[dict] = []
    session_meta: dict = {}
    events: list[dict] = []
    first_ts = last_ts = None
    # The first prompt decides whose session this is; later real prompts rule it out.
    first_prompt: dict | None = None
    later_real_prompts = 0
    reported_tokens: dict | None = None

    try:
        for event in ADAPTERS[chosen](p):
            events.append(event)
            counts[event["type"]] += 1
            if event["ts"]:
                first_ts = first_ts or event["ts"]
                last_ts = event["ts"]
            etype = event["type"]
            if etype == "session_start":
                session_meta = {k: v for k, v in event.items() if k not in {"seq", "type", "harness", "raw", "ts"}}
            elif etype == "model_response":
                tok = event["tokens"]
                for key in ("input_total", "input_fresh", "input_cached", "input_cache_write", "output"):
                    tokens[key] += tok[key] or 0
                tokens["reasoning"] += tok["reasoning"] or 0
                if event.get("totals_disagree"):
                    warnings.append(
                        {
                            "type": "harness_total_mismatch",
                            "seq": event["seq"],
                            "detail": "the record's own total_tokens does not equal input + output",
                        }
                    )
                if tok["input_fresh"] < 0:
                    warnings.append(
                        {
                            "type": "negative_fresh_input",
                            "seq": event["seq"],
                            "detail": "input_cached + input_cache_write exceeded input_total; "
                            "the harness's cache accounting is not a subset as assumed",
                        }
                    )
                if event.get("latency_seconds") is not None:
                    latencies.append(event["latency_seconds"])
                if event.get("model"):
                    models[event["model"]] += 1
            elif etype == "config_applied":
                # Model and effort are cost drivers, so they are tallied by the pair: the
                # same model at a different effort is a materially different run.
                settings[(event.get("model"), event.get("effort"))] += 1
                for name in event.get("disabled_plugins") or []:
                    disabled_plugins[name] += 1
            elif etype == "cost_report" and event.get("cumulative"):
                # Monotonic within a session, so the last one is the session total. Kept
                # apart from `tokens` above, which sums per-response records: these are
                # not the same quantity and adding them would count a session twice.
                reported_tokens = event
            elif etype == "user_prompt":
                if first_prompt is None:
                    first_prompt = event
                elif not event.get("interruption"):
                    later_real_prompts += 1
            elif etype == "tool_call":
                tools[event.get("tool") or "__unknown__"] += 1
                # Identical inputs to the same tool: the raw signal a repetition analyzer needs.
                repeats[(event.get("tool"), event.get("input_digest"))] += 1
            elif etype == "tool_result" and event.get("error"):
                tool_errors[event.get("tool") or "__unattributed__"] += 1
    except RuntimeError as exc:
        raise RuntimeError(str(exc)) from exc

    duplicate_calls = {
        f"{tool}:{digest}": n for (tool, digest), n in sorted(repeats.items(), key=lambda kv: -kv[1]) if n > 1
    }
    latencies.sort()
    origin = self_generated_verdict(
        (first_prompt or {}).get("produced_by"),
        (first_prompt or {}).get("signature_evidence"),
        counts["user_prompt"], later_real_prompts)
    return {
        "schema_version": EVENT_SCHEMA_VERSION,
        "mode": "normalize",
        "harness": chosen,
        "source": str(p),
        "read_only": True,
        "session": session_meta,
        "observed": {
            "events": len(events),
            "event_types": dict(sorted(counts.items())),
            "first_event_ts": first_ts,
            "last_event_ts": last_ts,
            "tokens": dict(sorted(tokens.items())),
            "models": dict(sorted(models.items())),
            "tool_calls": dict(tools.most_common()),
            "tool_errors": dict(tool_errors.most_common()),
            "latency_seconds": {
                "measured": len(latencies),
                "total": round(sum(latencies), 2),
                "max": latencies[-1] if latencies else None,
            },
            "repeated_identical_calls": duplicate_calls,
            # A harness-reported running total, which for most Codex versions is the only
            # token record there is. Separate from `tokens` because it is cumulative and
            # per-session, not summed per response — the two must never be added.
            "tokens_reported": ({
                "source": "event_msg/token_count",
                "cumulative": True,
                "breakdown_present": bool(reported_tokens.get("breakdown_present")),
                "context_window": reported_tokens.get("context_window"),
                **(reported_tokens.get("tokens") or {}),
            } if reported_tokens else None),
            # Codex records the configuration actually in effect; Claude Code does not, so
            # this is empty rather than zero for a Claude session.
            "config_applied": {
                f"{model or '?'}/{effort or '?'}": n
                for (model, effort), n in settings.most_common()
            },
            "disabled_plugins": dict(disabled_plugins.most_common()),
            # Set only when this tool wrote the session itself. Stage 6 measures a schema by
            # running `claude -p`, which the harness records like any other session, so
            # without this the normalized model counts the profiler's own exhaust as the
            # user's work — and anything built on it, `analyze` included, inherits that.
            "origin": origin,
        },
        "events": events,
        "integrity": {
            "warnings": warnings,
            "unlinked_tool_results": counts["tool_result"] - counts["tool_call"]
            if counts["tool_result"] > counts["tool_call"]
            else 0,
        },
        "profiler_cost": {
            "stage": "normalize",
            "bytes_read": p.stat().st_size,
            "events_emitted": len(events),
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
            "llm_input_tokens": 0,
            "llm_output_tokens": 0,
        },
        "notes": [
            "Events use a harness-independent vocabulary; raw record names appear only under "
            "each event's `raw.record`.",
            "input_total is the whole input side. Claude Code bills fresh/cached/cache-write "
            "separately, Codex reports one input figure that already contains the cached part; "
            "input_fresh + input_cached + input_cache_write == input_total holds for both.",
            "Tool inputs are represented by `input_digest` rather than content, because an "
            "argument can hold a credential. Equality of digests is enough to detect repetition.",
            "repeated_identical_calls counts identical (tool, input) pairs. Repetition is "
            "Observed; whether it was avoidable is not decided here.",
        ],
    }


def _codex_session_cwd(path: Path) -> str | None:
    """Read just the `session_meta` record to learn where a Codex session ran.

    Codex stores sessions by date, so this is the only way to attribute one to a project.
    Only the opening records are read, and decompression stops as soon as it is found.
    """
    try:
        for i, line in enumerate(_codex_lines(path)):
            if i > 8:
                break
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict) and obj.get("type") == "session_meta":
                payload = obj.get("payload")
                cwd = payload.get("cwd") if isinstance(payload, dict) else None
                return cwd if isinstance(cwd, str) else None
    except (OSError, RuntimeError):
        return None
    return None


def discover_codex_sessions(root: Path | None = None) -> list[dict]:
    """List Codex rollout transcripts, optionally only those whose cwd matches a root.

    Filtering happens here rather than after a limit is applied, so that asking for N
    sessions of a project returns N of *that project's* sessions.
    """
    codex_home = _user_tree("AGENT_PROFILER_CODEX_HOME", ".codex")
    found = []
    for entry in sorted(codex_home.glob(CODEX_SESSION_GLOB)):
        try:
            st = entry.stat()
        except OSError:
            continue
        cwd = None
        if root is not None:
            cwd = _codex_session_cwd(entry)
            if cwd is None or Path(cwd) != root:
                continue
        found.append(
            {
                "session_id": entry.name.split("-", 1)[-1].split(".")[0],
                "path": str(entry),
                "bytes": st.st_size,
                "modified": datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat().replace("+00:00", "Z"),
                "compressed": entry.suffix == ".zst",
                "cwd": cwd,
            }
        )
    found.sort(key=lambda s: s["modified"], reverse=True)
    return found


def normalize_root(root: str, limit: int | None = None, harness: str | None = None,
                   *, whole_environment: bool = False, use_cache: bool = True) -> dict:
    """Normalize sessions from every supported harness.

    With `whole_environment` the corpus is every project's transcripts rather than one
    root's, for both harnesses: Claude Code project keys are enumerated instead of
    derived from the root, and Codex rollouts stop being filtered on cwd.
    """
    if not root or not root.strip():
        raise ValueError("root must not be empty")
    base = Path(root).expanduser().resolve()
    if not base.is_dir():
        raise NotADirectoryError(f"root is not a directory: {base}")

    started = time.perf_counter()
    targets: list[tuple[str, str]] = []

    if harness in (None, "claude_code"):
        projects = _user_tree("AGENT_PROFILER_CLAUDE_HOME", ".claude") / "projects"
        dirs = (sorted(d for d in projects.glob("*") if d.is_dir()) if whole_environment
                else [projects / _encode_project_key(base)])
        entries = [p for d in dirs if d.is_dir() for p in d.glob("*.jsonl")]
        entries.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        targets += [("claude_code", str(p)) for p in entries]

    if harness in (None, "codex"):
        # Codex stores sessions by date, not by project, so the root is matched on cwd
        # before any limit is applied — unless the question is about the environment,
        # in which case there is nothing to match against.
        for entry in discover_codex_sessions(None if whole_environment else base):
            targets.append(("codex", entry["path"]))

    per_harness_limit = limit if limit and limit > 0 else None
    if per_harness_limit:
        trimmed: list[tuple[str, str]] = []
        seen: Counter = Counter()
        for h, path in targets:
            if seen[h] >= per_harness_limit:
                continue
            seen[h] += 1
            trimmed.append((h, path))
        targets = trimmed

    sessions = []
    failures = []
    totals: Counter = Counter()
    event_types: Counter = Counter()
    tools: Counter = Counter()
    by_harness: Counter = Counter()
    user_requests: Counter = Counter()
    self_requests: Counter = Counter()
    reported: Counter = Counter()
    token_coverage: Counter = Counter()
    self_sessions: Counter = Counter()
    produced_by: Counter = Counter()
    bytes_read = 0

    # Keyed per harness as well as per path, because a transcript's harness is detected
    # rather than given and the summary differs by adapter.
    cache = FileCache("normalize", enabled=use_cache)

    def summarize(path, h):
        full = normalize_session(path, h)
        # Only what this function reads is kept: the event stream is the large part and no
        # caller of `normalize_root` looks at it.
        return {
            "source": full["source"],
            "session": full["session"],
            "observed": full["observed"],
            "integrity": full["integrity"],
            "bytes_read": full["profiler_cost"]["bytes_read"],
        }

    for h, path in targets:
        try:
            result = cache.get_or_compute(Path(path), lambda p=path, hh=h: summarize(p, hh))
        except (OSError, ValueError, RuntimeError) as exc:
            failures.append({"path": path, "harness": h, "error": exc.__class__.__name__,
                             "detail": str(exc)[:160]})
            continue
        obs = result["observed"]
        # Discovery already filtered on cwd; this stays as a guard for explicit paths.
        if h == "codex" and not whole_environment:
            cwd = result["session"].get("cwd")
            if cwd and Path(cwd) != base:
                continue
        by_harness[h] += 1
        bytes_read += result["bytes_read"]
        for key, value in obs["tokens"].items():
            totals[key] += value
        for key, value in obs["event_types"].items():
            event_types[key] += value
        for key, value in obs["tool_calls"].items():
            tools[key] += value
        # A model response is one request to the API, which is the denominator an
        # always-on prefix cost has to be multiplied by. The profiler's own probe
        # sessions are requests too, so counting them would bill this tool's traffic to
        # the person being profiled.
        # Which kind of token evidence this session actually has. A figure built from the
        # sessions that happen to carry a record says nothing about the ones that do not,
        # and on this corpus that was 91% of them — so the denominator is reported.
        rep = obs.get("tokens_reported")
        per_response = (obs["tokens"].get("input_total") or 0) > 0
        if per_response:
            token_coverage["per_response"] += 1
        elif rep and rep.get("breakdown_present"):
            token_coverage["session_total_with_breakdown"] += 1
        elif rep:
            token_coverage["session_total_only"] += 1
        else:
            token_coverage["none"] += 1
        # Summed across sessions, never within one: each session's figure is already that
        # session's running total, so adding the records inside it would multiply it.
        if rep and not per_response:
            for key in ("input_tokens", "cached_input_tokens", "cache_write_input_tokens",
                        "output_tokens", "reasoning_output_tokens", "total_tokens"):
                reported[key] += rep.get(key) or 0

        origin = obs.get("origin") or {}
        responses = obs["event_types"].get("model_response", 0)
        if origin.get("self_generated"):
            self_sessions[h] += 1
            self_requests[h] += responses
            produced_by[origin.get("produced_by") or "unknown"] += 1
        else:
            user_requests[h] += responses
        sessions.append(
            {
                "harness": h,
                "source": result["source"],
                "session": result["session"],
                "observed": obs,
                "integrity": result["integrity"],
            }
        )

    cache.flush()

    return {
        "schema_version": EVENT_SCHEMA_VERSION,
        "mode": "normalize_root",
        "root": str(base),
        "scope": "whole_environment" if whole_environment else "project",
        "read_only": True,
        "observed": {
            "sessions_normalized": len(sessions),
            "sessions_by_harness": dict(sorted(by_harness.items())),
            "event_types": dict(sorted(event_types.items())),
            "tokens": dict(sorted(totals.items())),
            # Harness-reported session totals, for the sessions with no per-response
            # record. Kept in its own block because it is a different measurement: not
            # per request, and on Codex 0.166+ carrying no breakdown at all. Adding it to
            # `tokens` above would merge two quantities whose identity cannot be checked.
            "tokens_reported": dict(sorted(reported.items())),
            "token_coverage": dict(sorted(token_coverage.items())),
            "tool_calls": dict(tools.most_common(25)),
            # The denominator for any always-on cost: one model response is one request.
            # Split by who sent it, because an always-on prefix is carried by the user's
            # requests and this tool's probes alike, and only the first is the user's bill.
            "requests": {
                "by_harness": dict(sorted(user_requests.items())),
                "total": sum(user_requests.values()),
                "self_generated": dict(sorted(self_requests.items())),
            },
            "self_generated": {
                "sessions": sum(self_sessions.values()),
                "requests": sum(self_requests.values()),
                "by": dict(produced_by.most_common()),
                "note": "Sessions this tool wrote itself, recognised by the prompt the "
                        "profiler sent. They are excluded from `requests` above. The token "
                        "and event totals in this block still include them — on the real "
                        "corpus that is 0.07% of input tokens against 1.5% of requests, so "
                        "the request denominator is where it actually mattered.",
            },
        },
        "sessions": sessions,
        "failures": failures,
        "profiler_cost": {
            "stage": "normalize_root",
            "sessions": len(sessions),
            "bytes_read": bytes_read,
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
            "llm_input_tokens": 0,
            "llm_output_tokens": 0,
        },
        # Never silent: a reader can always tell a reused summary from a fresh parse, and
        # `--no-cache` reparses everything.
        # Never silent: a reader can always tell a reused summary from a fresh parse, and
        # `--no-cache` re-reads everything.
        "reuse": {"enabled": cache.enabled,
                  "sessions_reused": cache.reused,
                  "sessions_parsed": cache.computed,
                  "keyed_on": "transcript bytes, package source and path"},
        "notes": [
            "One event vocabulary spans both harnesses; per-harness quirks stay in the adapters.",
            "Codex transcripts are stored by date rather than by project, so they are matched "
            "to a root by the session's recorded cwd.",
            "Counts are Observed. No judgment of value or avoidability is made here.",
        ],
    }


def write_events(result: dict, destination: str) -> dict:
    """Write a normalized session's events as JSONL, keeping the summary small."""
    out = Path(destination).expanduser()
    with out.open("w", encoding="utf-8") as fh:
        for event in result.get("events", []):
            fh.write(json.dumps(event, ensure_ascii=False) + "\n")
    return {"events_written": len(result.get("events", [])), "path": str(out)}
