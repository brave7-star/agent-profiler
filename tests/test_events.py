"""Regression suite for the normalized event model.

The point of Stage 3 is that later stages never see a raw format. These tests therefore
assert two things above all: both adapters produce the *same* vocabulary, and the token
invariant holds even though the two harnesses define their token fields differently.
"""

import json
import os
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path

from agent_profiler import core
from agent_profiler import events as ev
from agent_profiler.cli import main as cli_main
from agent_profiler.events import (
    EVENT_TYPES,
    detect_harness,
    normalize_root,
    normalize_session,
    write_events,
)

CLAUDE_SESSION = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"


def _claude_records():
    tool_block = {"type": "tool_use", "id": "toolu_A", "name": "Read", "input": {"file_path": "/proj/a.py"}}
    base = {"sessionId": CLAUDE_SESSION, "version": "2.1.325", "cwd": "/proj", "gitBranch": "main"}
    def rec(**kw):
        d = dict(base)
        d.update(kw)
        return d
    # `output_tokens_details` is on essentially every real record, and it is what marks the
    # usage as settled rather than mid-stream. A fixture without it is not the real shape.
    usage = {"input_tokens": 10, "cache_read_input_tokens": 1000,
             "cache_creation_input_tokens": 100, "output_tokens": 50,
             "output_tokens_details": {"thinking_tokens": 0}}
    msg = {"id": "msg_1", "role": "assistant", "model": "claude-opus-5-5",
           "stop_reason": "tool_use", "usage": usage}
    return [
        rec(type="api-request", id="apireq-1", timestamp="2026-11-18T10:00:00.000Z"),
        rec(type="user", message={"role": "user", "content": [{"type": "text", "text": "go"}]},
            timestamp="2026-11-18T10:00:00.500Z"),
        # Same response, two records, usage repeated: must yield ONE model_response.
        rec(type="assistant", message=dict(msg, content=[{"type": "thinking", "thinking": "..."}]),
            apiBlockIndex=0, requestRef="apireq-1", timestamp="2026-11-18T10:00:05.000Z"),
        rec(type="assistant", message=dict(msg, content=[tool_block]),
            apiBlockIndex=1, requestRef="apireq-1", timestamp="2026-11-18T10:00:05.000Z"),
        # Replay of the same tool_use: must not become a second tool_call.
        rec(type="api-request-blob", message={"role": "assistant", "content": [tool_block]}),
        rec(type="user", message={"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "toolu_A", "is_error": True, "content": "nope"}]},
            timestamp="2026-11-18T10:00:06.000Z"),
        rec(type="compact-boundary", timestamp="2026-11-18T10:00:07.000Z"),
        rec(type="cost-state", totalCostUSD=0.25, totalAPIDuration=5000, totalToolDuration=10,
            startTime=1794996000000, modelUsage={"claude-opus-5-5": {"inputTokens": 10}}),
    ]


def _codex_records():
    """Codex's own shape: payload-wrapped, cached tokens *inside* input_tokens."""
    return [
        {"timestamp": "2026-11-18T10:00:00.000Z", "type": "session_meta",
         "payload": {"session_id": "cx-1", "cwd": "/proj", "cli_version": "0.162.0",
                     "model_provider": "openai"}},
        {"timestamp": "2026-11-18T10:00:01.000Z", "type": "response_item",
         "payload": {"type": "message", "role": "user",
                     "content": [{"type": "input_text", "text": "go"}]}},
        {"timestamp": "2026-11-18T10:00:02.000Z", "type": "response_item",
         "payload": {"type": "custom_tool_call", "call_id": "call_1", "name": "exec",
                     "input": "text(await tools.exec_command({cmd:\"ls\"}))"}},
        {"timestamp": "2026-11-18T10:00:03.000Z", "type": "response_item",
         "payload": {"type": "custom_tool_call_output", "call_id": "call_1",
                     "output": [{"type": "input_text", "text": "{\"exit_code\":1,\"output\":\"err\"}"}]}},
        {"timestamp": "2026-11-18T10:00:04.000Z", "type": "response_item",
         "payload": {"type": "function_call", "call_id": "call_2", "name": "apply_patch",
                     "arguments": "{\"path\": \"/proj/a.py\"}"}},
        {"timestamp": "2026-11-18T10:00:05.000Z", "type": "response_item",
         "payload": {"type": "function_call_output", "call_id": "call_2",
                     "output": [{"type": "input_text", "text": "{\"exit_code\":0}"}]}},
        # input_tokens (1110) already contains cached (1000).
        {"timestamp": "2026-11-18T10:00:06.000Z", "type": "token_usage_record",
         "payload": {"response_id": "resp_1", "usage": {
             "input_tokens": 1110, "cached_input_tokens": 1000, "cache_write_input_tokens": 0,
             "output_tokens": 50, "reasoning_output_tokens": 20, "total_tokens": 1160},
             "turn_token_usage": {"input_tokens": 99999, "output_tokens": 99999, "total_tokens": 199998}}},
        # Cumulative session figure: must never be summed into totals.
        {"timestamp": "2026-11-18T10:00:07.000Z", "type": "event_msg",
         "payload": {"type": "token_count", "info": {"total_token_usage": {
             "input_tokens": 888888, "output_tokens": 888888, "total_tokens": 1777776}}}},
        {"timestamp": "2026-11-18T10:00:08.000Z", "type": "compacted",
         "payload": {"message": "", "replacement_history": []}},
        {"timestamp": "2026-11-18T10:00:09.000Z", "type": "event_msg",
         "payload": {"type": "turn_aborted"}},
    ]


@contextmanager
def _env(*, claude=True, codex=True, codex_records=None, claude_records=None):
    with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as h:
        root, home = Path(d), Path(h)
        paths = {}
        if claude:
            key = str(root.resolve()).replace("/", "-").replace(".", "-")
            cdir = home / ".claude" / "projects" / key
            cdir.mkdir(parents=True)
            p = cdir / f"{CLAUDE_SESSION}.jsonl"
            p.write_text("\n".join(json.dumps(r) for r in (claude_records or _claude_records()))
                         + "\n", encoding="utf-8")
            paths["claude"] = p
        if codex:
            xdir = home / ".codex" / "sessions" / "2026" / "11" / "18"
            xdir.mkdir(parents=True)
            # cwd must match the root for root-scoped normalization to pick it up.
            records = codex_records or _codex_records()
            records[0]["payload"]["cwd"] = str(root.resolve())
            p = xdir / "rollout-2026-11-18T10-00-00-cx-1.jsonl"
            p.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
            paths["codex"] = p
        saved = {k: os.environ.get(k) for k in
                 ("AGENT_PROFILER_HOME", "AGENT_PROFILER_CLAUDE_HOME", "AGENT_PROFILER_CODEX_HOME")}
        os.environ["AGENT_PROFILER_HOME"] = str(home)
        os.environ["AGENT_PROFILER_CLAUDE_HOME"] = str(home / ".claude")
        os.environ["AGENT_PROFILER_CODEX_HOME"] = str(home / ".codex")
        try:
            yield root, paths
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v


def _by_type(result, etype):
    return [e for e in result["events"] if e["type"] == etype]


def test_both_adapters_share_one_vocabulary():
    """The whole point of the stage: no raw record name reaches the event type."""
    with _env() as (_root, paths):
        cc = normalize_session(str(paths["claude"]))
        cx = normalize_session(str(paths["codex"]))
        for result in (cc, cx):
            assert set(e["type"] for e in result["events"]) <= set(EVENT_TYPES)
        shared = {"session_start", "user_prompt", "model_response", "tool_call",
                  "tool_result", "context_compaction"}
        assert shared <= set(e["type"] for e in cc["events"])
        assert shared <= set(e["type"] for e in cx["events"])
        # Summary keys are identical across harnesses, so consumers need no special-casing.
        assert set(cc["observed"]) == set(cx["observed"])
        assert set(cc["observed"]["tokens"]) == set(cx["observed"]["tokens"])


def test_token_invariant_holds_for_both_despite_different_semantics():
    """Claude bills cached separately; Codex folds it into input_tokens."""
    with _env() as (_root, paths):
        cc = normalize_session(str(paths["claude"]))["observed"]["tokens"]
        cx = normalize_session(str(paths["codex"]))["observed"]["tokens"]
        for t in (cc, cx):
            assert t["input_fresh"] + t["input_cached"] + t["input_cache_write"] == t["input_total"]
        # Claude: 10 fresh + 1000 cached + 100 written are three separate charges.
        assert cc["input_total"] == 1110 and cc["input_fresh"] == 10 and cc["input_cached"] == 1000
        # Codex: input_tokens 1110 already contains the 1000 cached, so fresh is 110.
        assert cx["input_total"] == 1110 and cx["input_fresh"] == 110 and cx["input_cached"] == 1000
        # Mapping Codex's cached onto Anthropic's semantics would have given 2110.
        assert cx["input_total"] != 2110
        assert cx["reasoning"] == 20 and cc["reasoning"] == 0


def test_cumulative_codex_figures_are_never_summed():
    """turn_token_usage and token_count are cumulative; summing them explodes totals."""
    with _env() as (_root, paths):
        obs = normalize_session(str(paths["codex"]))["observed"]
        assert obs["tokens"]["input_total"] == 1110
        assert obs["tokens"]["output"] == 50
        assert obs["event_types"]["model_response"] == 1


def test_claude_block_split_and_replays_collapse():
    with _env() as (_root, paths):
        r = normalize_session(str(paths["claude"]))
        assert len(_by_type(r, "model_response")) == 1
        assert len(_by_type(r, "tool_call")) == 1
        assert r["observed"]["tokens"]["input_total"] == 1110


def _streamed_records(*copies):
    """One response written as several records, each copy's usage given as passed."""
    base = {"sessionId": CLAUDE_SESSION, "version": "2.1.232", "cwd": "/proj"}
    out = [dict(base, type="user", timestamp="2026-11-18T10:00:00.000Z",
                message={"role": "user", "content": [{"type": "text", "text": "go"}]})]
    for index, usage in enumerate(copies):
        out.append(dict(base, type="assistant", apiBlockIndex=index,
                        timestamp="2026-11-18T10:00:0%d.000Z" % (index + 1),
                        message={"id": "msg_s", "role": "assistant", "stop_reason": "end_turn",
                                 "model": "claude-opus-5-5", "usage": usage}))
    return out


def _partial(output):
    return {"input_tokens": 2, "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 500, "output_tokens": output}


def _settled(output):
    return dict(_partial(output), output_tokens_details={"thinking_tokens": 0},
                iterations=[{"type": "message", "output_tokens": output}])


def test_a_response_recorded_while_it_streams_reports_the_settled_total():
    """Counting once per message id used to mean counting the *first* copy.

    A real transcript wrote one response three times as it streamed, each copy carrying a
    larger `output_tokens` than the last, so counting-once reported a figure two orders of
    magnitude under the truth. The values below are constructed to that shape.
    """
    with _env(codex=False, claude_records=_streamed_records(
            _partial(1), _partial(3), _settled(300))) as (_root, paths):
        r = normalize_session(str(paths["claude"]))
        responses = _by_type(r, "model_response")
        assert len(responses) == 1
        assert responses[0]["tokens"]["output"] == 300
        assert r["observed"]["tokens"]["output"] == 300


def test_a_response_whose_copies_never_settle_is_still_counted():
    """A partial figure is the only figure such a transcript has; dropping it loses the
    response, and an absent response is a worse error than an understated one."""
    with _env(codex=False, claude_records=_streamed_records(
            _partial(7), _partial(7))) as (_root, paths):
        r = normalize_session(str(paths["claude"]))
        responses = _by_type(r, "model_response")
        assert len(responses) == 1
        assert responses[0]["tokens"]["output"] == 7


def test_a_held_response_keeps_its_place_in_the_record_order():
    """Held copies are flushed when the next response starts, not at end of file, so `seq`
    stays in the order the records were written."""
    records = _streamed_records(_partial(7)) + _streamed_records(_settled(9))
    records[-1]["message"]["id"] = "msg_t"
    records[-1]["timestamp"] = "2026-11-18T10:00:30.000Z"
    with _env(codex=False, claude_records=records) as (_root, paths):
        r = normalize_session(str(paths["claude"]))
        assert [e["response_id"] for e in _by_type(r, "model_response")] == ["msg_s", "msg_t"]
        stamps = [e["ts"] for e in r["events"] if e["ts"]]
        assert stamps == sorted(stamps)


def test_every_event_carries_a_raw_reference():
    """A number nobody can trace back to bytes is not evidence."""
    with _env() as (_root, paths):
        for key in ("claude", "codex"):
            r = normalize_session(str(paths[key]))
            for event in r["events"]:
                raw = event["raw"]
                assert raw["source"].endswith((".jsonl", ".zst"))
                assert isinstance(raw["line"], int) and raw["line"] >= 1
                assert raw["record"]
                assert isinstance(event["seq"], int)
            seqs = [e["seq"] for e in r["events"]]
            assert seqs == sorted(seqs) and len(set(seqs)) == len(seqs)


def test_tool_inputs_are_digested_not_disclosed():
    """Arguments can hold credentials, so content must not reach the output."""
    secret = "Bearer SUPER-SECRET-TOKEN-VALUE"
    records = _claude_records()
    records[3]["message"]["content"] = [
        {"type": "tool_use", "id": "toolu_S", "name": "Bash",
         "input": {"command": f"curl -H 'Authorization: {secret}' https://x.invalid"}}
    ]
    with _env() as (_root, paths):
        paths["claude"].write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
        r = normalize_session(str(paths["claude"]))
        blob = json.dumps(r)
        assert secret not in blob
        assert "x.invalid" not in blob
        call = _by_type(r, "tool_call")[0]
        assert len(call["input_digest"]) == 16
        assert call["target"] is None  # a command has no safe target to carry


def test_safe_targets_are_kept_for_path_bearing_tools():
    with _env() as (_root, paths):
        cc = _by_type(normalize_session(str(paths["claude"])), "tool_call")[0]
        assert cc["tool"] == "Read" and cc["target"] == "/proj/a.py"
        # Codex function_call arguments are JSON, so a path inside them is recovered too.
        calls = _by_type(normalize_session(str(paths["codex"])), "tool_call")
        patch = [c for c in calls if c["tool"] == "apply_patch"][0]
        assert patch["target"] == "/proj/a.py"


def test_identical_calls_are_counted_without_content():
    records = _claude_records()
    dup = {"type": "tool_use", "id": "toolu_B", "name": "Read", "input": {"file_path": "/proj/a.py"}}
    records.insert(4, dict(records[3], message=dict(records[3]["message"], id="msg_2", content=[dup])))
    with _env() as (_root, paths):
        paths["claude"].write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
        obs = normalize_session(str(paths["claude"]))["observed"]
        repeats = obs["repeated_identical_calls"]
        assert len(repeats) == 1
        assert next(iter(repeats.values())) == 2
        assert next(iter(repeats)).startswith("Read:")


def test_tool_errors_normalize_across_harnesses():
    """Claude flags is_error; Codex only exposes a non-zero exit_code."""
    with _env() as (_root, paths):
        cc = normalize_session(str(paths["claude"]))
        assert cc["observed"]["tool_errors"] == {"Read": 1}
        cx = normalize_session(str(paths["codex"]))
        assert cx["observed"]["tool_errors"] == {"exec": 1}
        results = _by_type(cx, "tool_result")
        assert {r["tool"]: r["ok"] for r in results} == {"exec": False, "apply_patch": True}


def test_latency_present_for_claude_absent_for_codex():
    """Conclusion strength: Codex records give no request-start, so latency stays null."""
    with _env() as (_root, paths):
        cc = normalize_session(str(paths["claude"]))
        assert _by_type(cc, "model_response")[0]["latency_seconds"] == 5.0
        assert cc["observed"]["latency_seconds"]["measured"] == 1
        cx = normalize_session(str(paths["codex"]))
        assert _by_type(cx, "model_response")[0]["latency_seconds"] is None
        assert cx["observed"]["latency_seconds"]["measured"] == 0


def test_harness_is_detected_from_content():
    with _env() as (_root, paths):
        assert detect_harness(paths["claude"]) == "claude_code"
        assert detect_harness(paths["codex"]) == "codex"


def test_integrity_warns_when_cache_is_not_a_subset():
    """If a harness ever reports cached > input, say so instead of printing a negative."""
    records = _codex_records()
    records[6]["payload"]["usage"].update({"input_tokens": 100, "cached_input_tokens": 500})
    with _env() as (_root, paths):
        paths["codex"].write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
        r = normalize_session(str(paths["codex"]))
        kinds = {w["type"] for w in r["integrity"]["warnings"]}
        assert "negative_fresh_input" in kinds
        assert "harness_total_mismatch" in kinds


def test_empty_and_broken_transcripts_keep_a_stable_shape():
    with _env(codex=False) as (_root, paths):
        paths["claude"].write_text("\n{bad json\n[]\n\n", encoding="utf-8")
        r = normalize_session(str(paths["claude"]))
        assert r["events"] == []
        assert r["observed"]["events"] == 0
        # Stable keys even with nothing to report.
        assert set(r["observed"]["tokens"]) == {
            "input_total", "input_fresh", "input_cached", "input_cache_write", "output", "reasoning"}
        assert set(r["observed"]["event_types"]) == set(EVENT_TYPES)
        assert r["observed"]["tokens"]["input_total"] == 0


def test_root_normalization_spans_both_harnesses():
    with _env() as (root, _paths):
        r = normalize_root(str(root))
        assert r["observed"]["sessions_by_harness"] == {"claude_code": 1, "codex": 1}
        assert r["observed"]["sessions_normalized"] == 2
        # Totals add across harnesses because the token model is common.
        assert r["observed"]["tokens"]["input_total"] == 2220
        assert r["observed"]["event_types"]["tool_call"] == 3
        assert r["failures"] == []


def test_codex_sessions_are_matched_to_a_root_by_cwd():
    """Codex stores by date, not by project, so an unrelated cwd must be excluded."""
    records = _codex_records()
    records[0]["payload"]["cwd"] = "/somewhere/else"
    with _env() as (root, paths):
        paths["codex"].write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
        r = normalize_root(str(root))
        assert r["observed"]["sessions_by_harness"] == {"claude_code": 1}


def test_events_can_be_streamed_to_jsonl():
    with _env(codex=False) as (_root, paths):
        r = normalize_session(str(paths["claude"]))
        with tempfile.TemporaryDirectory() as out:
            dest = Path(out) / "events.jsonl"
            info = write_events(r, str(dest))
            lines = dest.read_text(encoding="utf-8").strip().splitlines()
            assert info["events_written"] == len(r["events"]) == len(lines)
            assert all(json.loads(line)["type"] in EVENT_TYPES for line in lines)


def test_compressed_codex_transcripts_are_read_when_zstd_exists():
    if not _which("zstd"):
        print("     (skipped: zstd not installed)")
        return
    with _env() as (_root, paths):
        plain = paths["codex"]
        subprocess.run(["zstd", "-q", "-f", str(plain), "-o", str(plain) + ".zst"], check=True)
        r = normalize_session(str(plain) + ".zst")
        assert r["harness"] == "codex"
        assert r["observed"]["tokens"]["input_total"] == 1110


def _which(name):
    from shutil import which
    return which(name)


def test_cli_reports_bad_input_as_exit_2():
    with _env() as (root, paths):
        assert cli_main(["normalize", "--session", str(paths["claude"]), "--output", os.devnull]) == 0
        assert cli_main(["normalize", "--root", str(root), "--output", os.devnull]) == 0
        assert cli_main(["normalize", "--root", ""]) == 2
        assert cli_main(["normalize", "--session", str(root / "missing.jsonl")]) == 2
        assert cli_main(["normalize", "--session", str(root)]) == 2



def test_codex_config_applied_captures_model_and_effort():
    """Audited last: turn_context and thread_settings_applied carry the configuration that
    was actually in effect. They do not name an instruction file, so AGENTS.md stays
    unproven — but model and reasoning effort are direct cost drivers, and 30% of real
    turns ran at xhigh.

    The two records name themselves at different levels, and this fixture used to get
    `turn_context` wrong — it wrote the name inside `payload`, under a `response_item`
    envelope, a shape that appears nowhere on disk. The test passed and the real records
    were dropped: every one of them, carrying exactly the model and effort this claims to
    capture. A fixture that disagrees with the machine proves nothing, so the shape below
    is the one `~/.codex/sessions` actually holds — type at the top, no `type` in payload.
    """
    records = _codex_records() + [
        {"timestamp": "2026-11-18T10:00:10.000Z", "ordinal": 13, "type": "turn_context",
         "payload": {"turn_id": "t1", "cwd": "/proj",
                     "model": "gpt-5.6-sol", "effort": "xhigh",
                     "approval_policy": "on-request",
                     "sandbox_policy": {"type": "workspace-write"},
                     "personality": "pragmatic"}},
        {"timestamp": "2026-11-18T10:00:11.000Z", "type": "event_msg",
         "payload": {"type": "thread_settings_applied", "thread_settings": {
             "model": "gpt-5.6-sol", "reasoning_effort": "low",
             "approval_policy": "on-request", "personality": "pragmatic",
             "disabled_plugin_ids": ["some-plugin"]}}},
    ]
    with _env(claude=False, codex=True, codex_records=records) as (_root, paths):
        result = normalize_session(str(paths["codex"]), "codex")
        observed = result["observed"]
    assert observed["event_types"]["config_applied"] == 2
    assert observed["config_applied"] == {"gpt-5.6-sol/xhigh": 1, "gpt-5.6-sol/low": 1}
    # Declared-but-switched-off is a real Installed-but-not-active signal.
    assert observed["disabled_plugins"] == {"some-plugin": 1}
    # Per-turn and per-thread must stay distinguishable: the scope label read the wrong
    # level too, so every turn record would have been filed as a thread setting.
    scopes = sorted(e["scope"] for e in result["events"] if e["type"] == "config_applied")
    assert scopes == ["thread", "turn"]


def test_a_claude_session_has_no_config_applied_rather_than_a_zero():
    """Claude Code does not record this, so the field must be empty, not a fabricated 0."""
    with _env(claude=True, codex=False) as (_root, paths):
        observed = normalize_session(str(paths["claude"]), "claude_code")["observed"]
    assert observed["config_applied"] == {}
    assert observed["event_types"]["config_applied"] == 0


def test_whole_environment_normalization_covers_both_harnesses_beyond_the_root():
    """Root-scoped normalization keys Claude Code off the root's project key and filters
    Codex rollouts on cwd. Neither is right for a question about the environment: on the
    real machine the root-scoped Codex corpus was empty and the global one was 640."""
    with _env() as (root, paths):
        # A second project key, and a Codex rollout recorded in a different directory.
        home = Path(os.environ["AGENT_PROFILER_CLAUDE_HOME"])
        other = home / "projects" / "-somewhere-else"
        other.mkdir(parents=True)
        (other / f"{CLAUDE_SESSION}.jsonl").write_text(
            "\n".join(json.dumps(r) for r in _claude_records()) + "\n", encoding="utf-8")
        elsewhere = [json.loads(json.dumps(r)) for r in _codex_records()]
        elsewhere[0]["payload"]["cwd"] = "/some/other/project"
        Path(paths["codex"]).parent.joinpath("rollout-2026-11-18T11-00-00-cx-2.jsonl").write_text(
            "\n".join(json.dumps(r) for r in elsewhere) + "\n", encoding="utf-8")

        scoped = normalize_root(str(root))
        whole = normalize_root(str(root), whole_environment=True)

    assert scoped["scope"] == "project" and whole["scope"] == "whole_environment"
    by_scoped = scoped["observed"]["sessions_by_harness"]
    by_whole = whole["observed"]["sessions_by_harness"]
    # The root's own session in each harness, and nothing from elsewhere.
    assert by_scoped == {"claude_code": 1, "codex": 1}
    # Both the other project key and the other directory's rollout, and no double counting.
    assert by_whole == {"claude_code": 2, "codex": 2}


def test_whole_environment_analysis_compares_against_a_matching_inventory():
    """Installed -> Used findings compare declarations to traced harnesses, so the scan has
    to be scoped like the traces. A global corpus checked against a root-scoped inventory
    would judge usage of servers it never looked for."""
    from agent_profiler.analyze import analyze_root

    with _env() as (root, _paths):
        scoped = analyze_root(str(root))
        whole = analyze_root(str(root), whole_environment=True)

    assert scoped["scope"] == "project" and whole["scope"] == "whole_environment"
    assert whole["observed"]["sessions_analyzed"] >= scoped["observed"]["sessions_analyzed"]
    assert whole["profiler_cost"]["llm_input_tokens"] == 0


def test_host_skills_carries_a_recorded_listing_cost():
    """`world_state` was written off as "a workspace snapshot, deliberately unused".

    It carries the one thing the project otherwise pays a paired probe to estimate: the
    skill listing's token cost, measured and recorded by the harness. On the real corpus
    these come back as the two listing sizes the fixture carries.
    """
    from agent_profiler.events import _host_skills_load

    payload = {"state": {"host_skills": {
        "version": 1, "instructionsSent": True, "omitDescriptions": False,
        "usage": {"tokens": 2100, "characters": 7200},
        "roots": {"r0": "/h/.codex/skills", "r1": "/h/.agents/skills"},
        "entries": {
            '["host","host","/h/.codex/skills/fourth-skill/SKILL.md"]':
                {"fingerprint": "a", "shown": True},
            '["host","host","/h/.agents/skills/third-skill/SKILL.md"]':
                {"fingerprint": "b", "shown": True},
        },
    }}}
    load = _host_skills_load(payload)
    assert load["load_kind"] == "skill_listing"
    assert load["tokens"] == 2100 and load["chars"] == 7200
    assert load["components"] == ["fourth-skill", "third-skill"]
    # One block covering many skills: shared, and never divided among them.
    assert load["shared"] is True
    assert load["depth"] == "description_only"
    # The roots are what exposed a skills directory the scanner was not looking at.
    assert "/h/.agents/skills" in load["roots"]


def test_host_skills_without_entries_is_not_an_event():
    """An empty snapshot must not become a zero-cost load event."""
    from agent_profiler.events import _host_skills_load

    assert _host_skills_load({"state": {"host_skills": {"entries": {}}}}) is None
    assert _host_skills_load({"state": {}}) is None
    assert _host_skills_load({}) is None


def _exec_call(command, call="call_x"):
    """Codex's `exec`: the input is a JS program, not a command."""
    return {"timestamp": "2026-11-18T10:00:02.000Z", "type": "response_item",
            "payload": {"type": "custom_tool_call", "call_id": call, "name": "exec",
                        "input": "const r = await tools.exec_command({cmd:\""
                                 + command + "\",workdir:\"/proj\"}); text(r.output)"}}


def test_codex_exec_is_recognised_as_a_shell_tool():
    """Real defect: `exec` was missing from SHELL_TOOLS while being the overwhelming majority
    of Codex
    tool calls, so every shell-based detector was blind to almost everything Codex does.
    Its input is a JavaScript program wrapping the command, which is why the raw string is
    searched rather than a `cmd` key read."""
    from agent_profiler.traces import _skill_body_read, _skill_body_paths

    call = _exec_call("cat /home/u/.codex/skills/fourth-skill/SKILL.md")["payload"]
    assert _skill_body_read("exec", call["input"]) == "fourth-skill"
    assert _skill_body_paths("exec", call["input"]) == ("/home/u/.codex/skills/fourth-skill/SKILL.md",)

    # A relative path is dropped: resolving it would need a cwd guess, and a wrong guess
    # credits the use to the wrong component.
    assert _skill_body_paths("exec", "cat skills/fourth-skill/SKILL.md") == ()


def test_codex_skill_use_is_attributed_by_installed_path_not_by_name():
    """A read of a vendor-bundled `pdf` skill must not be credited to a user skill of the
    same name. On a real corpus most candidate reads were `.codex/plugins/cache/...`
    bundles and a copy under `~/Downloads`, none of which are the user's components."""
    from agent_profiler.events import codex_component_usage

    mine = "/home/u/.codex/skills/fourth-skill/SKILL.md"
    records = [_codex_records()[0],
               _exec_call(f"cat {mine}", "c1"),
               _exec_call("cat /home/u/.codex/plugins/cache/openai-bundled/fourth-skill/SKILL.md", "c2"),
               _exec_call("cat /home/u/Downloads/1/skills/fourth-skill/SKILL.md", "c3")]
    with _env(claude=False, codex_records=records) as (root, _paths):
        usage = codex_component_usage({"user/codex/fourth-skill": mine}, None)
    assert usage["reads"] == {"user/codex/fourth-skill": 1}
    assert usage["sessions"] == {"user/codex/fourth-skill": 1}


def test_a_session_that_edits_a_skill_is_authoring_it_not_using_it():
    """The project's recorded 544-false-positive case. Editing an *installed* skill reads
    the installed path, so the path rule alone cannot separate them; a write to the same
    file in the same session can. On the real corpus this is decisive: `fifth-skill`'s only
    session wrote to it, while `fourth-skill` was read and never written."""
    from agent_profiler.events import codex_component_usage

    mine = "/home/u/.codex/skills/fifth-skill/SKILL.md"
    records = [_codex_records()[0],
               _exec_call(f"cat {mine}", "c1"),
               _exec_call(f"cat > {mine} <<'EOF'", "c2")]
    with _env(claude=False, codex_records=records) as (root, _paths):
        usage = codex_component_usage({"user/codex/fifth-skill": mine}, None)
    assert usage["reads"] == {}
    assert usage["authoring_sessions"] == {"user/codex/fifth-skill": 1}

    # An apply_patch to the file counts as a write even with no shell redirection.
    patched = [_codex_records()[0],
               _exec_call(f"cat {mine}", "c1"),
               {"timestamp": "2026-11-18T10:00:04.000Z", "type": "response_item",
                "payload": {"type": "function_call", "call_id": "c2", "name": "apply_patch",
                            "arguments": json.dumps({"path": mine, "cmd": mine})}}]
    with _env(claude=False, codex_records=patched) as (root, _paths):
        assert codex_component_usage({"user/codex/fifth-skill": mine}, None)["reads"] == {}


def test_the_codex_skill_listing_proves_loaded_per_file():
    """`loaded` said 「declared for codex, whose traces were not analyzed」 for eight
    components while `world_state.host_skills.entries` named each one's exact file. The
    extractor was keeping only the name, which cannot attribute: `demo-skill` exists under
    two roots here, so a name proves nothing about which was listed."""
    from agent_profiler.events import _host_skills_load
    from agent_profiler.score import _entity_loaded

    mine = "/home/u/.codex/skills/fourth-skill/SKILL.md"
    other = "/home/u/.agents/skills/fourth-skill/SKILL.md"
    load = _host_skills_load({"state": {"host_skills": {
        "entries": {json.dumps(["host", "host", mine]): {"shown": True}},
        "usage": {"tokens": 2100, "characters": 7200}, "roots": {"a": "/home/u/.codex/skills"}}}})
    assert load["component_paths"] == (lambda p: [p])(mine)
    assert load["components"] == ["fourth-skill"]

    listing = {"sessions_listing_path": {mine: 5}, "sessions_carrying_record": 5}
    link = _entity_loaded("fourth-skill", "skills", "codex", {}, set(), 0, listing, mine)
    assert link["status"] == "proven" and link["value"] == 5
    # The denominator is the sessions that carried a listing, not the whole corpus.
    assert "5 of the 5 session(s) that recorded one" in link["evidence"]

    # The same-named skill under the other root is not proven by this record.
    assert _entity_loaded("fourth-skill", "skills", "shared", {}, set(), 0,
                          listing, other)["status"] == "unproven"


def test_a_codex_mcp_call_inside_an_exec_program_is_a_call():
    """`used` said 「declared for codex, whose traces were not analyzed」 for two MCP
    servers while the corpus was full of `repl_server` calls. Codex emits no MCP tool call of
    its own — the model writes `tools.mcp__<server>__<tool>(...)` into the program it hands
    to `exec` — so the detector was looking for something that never exists."""
    from agent_profiler.traces import _mcp_servers_called

    program = 'const r = await tools.mcp__repl_server__js({code:"1"}); text(r)'
    assert _mcp_servers_called("exec", program) == ("repl_server",)

    # A mention is not a call: the model routinely filters the tool list by name.
    assert _mcp_servers_called(
        "exec", 'ALL_TOOLS.filter(x => /repl_server|browser/i.test(x.name))') == ()

    # Declared with a hyphen, addressed with an underscore.
    assert _mcp_servers_called("exec", "await tools.mcp__demo_server__get_task({})") \
        == ("demo_server",)


def test_a_zero_codex_mcp_count_is_only_evidence_because_positives_are_detected():
    """A zero from a blind detector is not a finding. This one is reported as proven only
    alongside the fact that other servers in the same corpus were detected."""
    from agent_profiler.score import build_chain

    scan = {"observed": {"components": {}}, "files": [],
            "mcp_servers": [{"name": "computer-use", "source": "user_codex_dir",
                             "scope": "user", "declared_in": "/x/config.toml"}]}
    traces = {"observed": {"sessions_collected": 3, "calls_by_tool": {}}}
    usage = {"sessions_scanned": 640, "mcp_calls": {"repl_server": 1665},
             "mcp_sessions": {"repl_server": 62}, "reads": {}, "sessions": {},
             "authoring_sessions": {}}
    link = build_chain(scan=scan, traces=traces, codex_usage=usage)[0]["links"]["used"]
    assert link["status"] == "proven" and link["value"] == 0
    assert "640" in link["evidence"] and "other servers are detected" in link["evidence"]

    # Without a Codex scan there is no detector, so the same zero stays unproven.
    bare = build_chain(scan=scan, traces=traces)[0]["links"]["used"]
    assert bare["status"] == "unproven" and bare["blocker"] == "harness_not_traced"


def test_a_by_name_invocation_is_not_attributed_to_one_of_two_installed_copies():
    """Real defect: `~/.agents/skills` and `~/.claude/skills` both hold `third-skill`, and
    its 6 invocations were credited in full to each — reporting 6 calls as 12. A `Skill`
    call carries only `{skill, args}` (checked against every such call in the corpus), and
    the harness's own listing shows each name once, having already deduplicated the roots.
    """
    from agent_profiler.score import _entity_used

    both = {"claude_code", "shared"}
    link = _entity_used("third-skill", "skills", "shared", {"third-skill": 6}, {"claude_code"},
                        255, None, "user/shared/third-skill", both)
    assert link["status"] == "partial" and link["value"] == 6
    assert "cannot be attributed" not in link["evidence"]  # it says what IS known first
    assert "unknown" in link["evidence"] and "2 roots" in link["evidence"]

    # Installed in one place only, the same count is a proof.
    alone = _entity_used("third-skill", "skills", "claude_code", {"third-skill": 6},
                         {"claude_code"}, 255, None, "user/claude_code/third-skill", None)
    assert alone["status"] == "proven" and alone["value"] == 6


def test_a_shared_skill_root_is_not_treated_as_an_untraced_harness():
    """`shared` marks `~/.agents/skills`, which both harnesses read — Codex's own listing
    names it as a root. Treating it as a harness awaiting traces made those components
    permanently unprovable on `used`, waiting for a harness that does not exist."""
    from agent_profiler.score import build_chain

    scan = {"observed": {"components": {"skills": {
                "entities": 1, "files": 1, "bytes": 1,
                "names": ["user/shared/solo"]}}},
            "files": [{"kind": "skills", "origin": "declared", "scope": "user",
                       "harness": "shared", "path": "/x/.agents/skills/solo/SKILL.md"}],
            "mcp_servers": []}
    traces = {"observed": {"sessions_collected": 10, "harnesses": ["claude_code"],
                           "components_invoked": {"solo": 4}}}
    link = build_chain(scan=scan, traces=traces)[0]["links"]["used"]
    assert link["status"] == "proven" and link["value"] == 4
    assert "traces were not analyzed" not in link["evidence"]


def test_codex_skill_use_is_never_reported_as_proof():
    """A read is weaker than an invocation and must stay `partial` however many there are:
    the project cannot fully separate using a skill from editing it."""
    from agent_profiler.score import _entity_used

    usage = {"reads": {"user/codex/fourth-skill": 99}, "sessions": {"user/codex/fourth-skill": 40},
             "authoring_sessions": {}}
    link = _entity_used("fourth-skill", "skills", "codex", {}, set(), 10,
                        usage, "user/codex/fourth-skill")
    assert link["status"] == "partial"
    assert "99" in link["evidence"] and "40" in link["evidence"]
    assert "not proof" in link["evidence"]


@contextmanager
def _cache_file():
    """A cache of its own, so no test reads or writes the real one."""
    with tempfile.TemporaryDirectory() as d:
        saved = os.environ.get("AGENT_PROFILER_CACHE")
        os.environ["AGENT_PROFILER_CACHE"] = str(Path(d) / "cache.json")
        try:
            yield Path(d) / "cache.json"
        finally:
            if saved is None:
                os.environ.pop("AGENT_PROFILER_CACHE", None)
            else:
                os.environ["AGENT_PROFILER_CACHE"] = saved


def _measurements(result):
    """Everything a reader would act on, with the self-reported fields removed."""
    return {"observed": result["observed"],
            "sessions": result["sessions"],
            "failures": result["failures"]}


def test_a_reused_summary_equals_the_one_it_replaces():
    """Reusing a summary is only sound if it is indistinguishable from parsing again. The
    timing and the reuse counters are expected to differ; nothing a reader acts on may."""
    with _cache_file(), _env() as (root, _paths):
        cold = normalize_root(str(root), whole_environment=True)
        warm = normalize_root(str(root), whole_environment=True)

    assert cold["reuse"]["sessions_parsed"] == 2 and cold["reuse"]["sessions_reused"] == 0
    assert warm["reuse"]["sessions_reused"] == 2 and warm["reuse"]["sessions_parsed"] == 0
    assert _measurements(cold) == _measurements(warm)


def test_a_same_size_edit_in_the_same_second_still_invalidates():
    """The exact failure this cache is keyed to avoid, and one this project paid for: a
    bytecode cache that validates on a source's size and whole-second mtime served stale
    code after a `rtype`/`ptype` edit, because the two are the same length and the write
    landed inside the same second. So the key is the transcript's bytes. Here the content
    changes, the length does not, and the mtime is forced back to what it was."""
    with _cache_file(), _env() as (root, paths):
        first = normalize_root(str(root), whole_environment=True)
        assert first["reuse"]["sessions_parsed"] == 2

        transcript = paths["codex"]
        before = transcript.read_text(encoding="utf-8")
        stat = transcript.stat()
        # `exec` -> `exeC`: the same number of bytes, a different tool name in the record.
        after = before.replace('"name": "exec"', '"name": "exeC"', 1)
        assert after != before and len(after) == len(before)
        transcript.write_text(after, encoding="utf-8")
        os.utime(transcript, (stat.st_atime, stat.st_mtime))
        assert transcript.stat().st_size == stat.st_size
        assert int(transcript.stat().st_mtime) == int(stat.st_mtime)

        second = normalize_root(str(root), whole_environment=True)

    # The Claude transcript is untouched and may be reused; the edited one may not.
    assert second["reuse"]["sessions_parsed"] >= 1
    reused_paths = second["reuse"]["sessions_reused"]
    assert reused_paths <= 1, "the edited transcript was served from the cache"


def test_editing_a_parser_invalidates_every_entry_without_a_version_bump():
    """A hand-maintained version constant is a constant someone eventually forgets, and the
    failure is a stale number presented as a measurement. The key carries a digest of the
    parser sources instead, so this needs no cooperation from whoever edits them."""
    with _cache_file(), _env() as (root, _paths):
        normalize_root(str(root), whole_environment=True)
        warm = normalize_root(str(root), whole_environment=True)
        assert warm["reuse"]["sessions_reused"] == 2

        saved = core._CODE_FINGERPRINT
        try:
            core._CODE_FINGERPRINT = "pretend-the-package-changed"
            after = normalize_root(str(root), whole_environment=True)
        finally:
            core._CODE_FINGERPRINT = saved

    assert after["reuse"]["sessions_reused"] == 0
    assert after["reuse"]["sessions_parsed"] == 2


def test_reuse_is_reported_and_can_be_switched_off():
    """A reused figure must never be indistinguishable from a fresh one in the output."""
    with _cache_file(), _env() as (root, _paths):
        normalize_root(str(root), whole_environment=True)
        off = normalize_root(str(root), whole_environment=True, use_cache=False)

    assert off["reuse"]["enabled"] is False
    assert off["reuse"]["sessions_reused"] == 0
    assert off["reuse"]["sessions_parsed"] == 2


def test_an_unwritable_cache_does_not_fail_the_run():
    """The cache is an optimization. A read-only home, a full disk or a corrupt file must
    cost speed and nothing else."""
    with _env() as (root, _paths):
        saved = os.environ.get("AGENT_PROFILER_CACHE")
        with tempfile.TemporaryDirectory() as d:
            broken = Path(d) / "cache.json"
            broken.write_text("{not json", encoding="utf-8")
            os.environ["AGENT_PROFILER_CACHE"] = str(broken)
            try:
                corrupt = normalize_root(str(root), whole_environment=True)
            finally:
                if saved is None:
                    os.environ.pop("AGENT_PROFILER_CACHE", None)
                else:
                    os.environ["AGENT_PROFILER_CACHE"] = saved
    assert corrupt["observed"]["sessions_normalized"] == 2
    assert corrupt["reuse"]["sessions_reused"] == 0


def test_the_codex_request_count_is_also_split_by_the_directory_it_ran_in():
    """A project-scoped AGENTS.md is carried only by requests made inside that project, so
    the requests have to be attributable to a directory. Without the split, the only
    multiplier on offer is the environment total, which charges one project's instructions
    to every request ever made — and the split must still sum to that total, or one of the
    two figures is wrong."""
    from agent_profiler.events import codex_shared_context

    with _env(claude=False, codex=True) as (root, _paths):
        out = codex_shared_context(None, use_cache=False)
    by_cwd = out["requests_by_cwd"]
    assert sum(by_cwd.values()) == out["requests"]
    # The fixture's session records its cwd, so the requests land under a real directory
    # rather than an empty key.
    assert list(by_cwd) == [str(root.resolve())]


def test_both_codex_rollups_come_from_one_pass_over_the_rollouts():
    """`report._collect` promises the deterministic stages run "once each", and collected
    the whole Codex corpus twice, at tens of seconds a pass. The two rollups now share a
    walk, so each must still equal what its own function returns alone."""
    from agent_profiler.events import (codex_component_usage, codex_rollups,
                                       codex_shared_context)

    skill = "/x/.codex/skills/fourth-skill/SKILL.md"
    records = _codex_records() + [_exec_call(f"cat {skill}", "c1")]
    with _env(claude=False, codex=True, codex_records=records) as (root, _paths):
        paths = {"user/codex/fourth-skill": skill}
        shared_alone = codex_shared_context(None)
        usage_alone = codex_component_usage(paths, None)
        shared_together, usage_together = codex_rollups(paths, None)

    # `reuse` is the one field expected to differ: the separate calls populate the cache
    # that the combined call then reuses. Everything a reader acts on must match.
    def measured(result):
        return {k: v for k, v in result.items() if k != "reuse"}

    assert measured(shared_alone) == measured(shared_together)
    assert measured(usage_alone) == measured(usage_together)
    # And each must still say what it reused, so no figure is silently second-hand.
    assert shared_together["reuse"]["sessions_reused"] == 1
    assert usage_together["reuse"]["sessions_reused"] == 1
    # And asking for no attribution must return nothing rather than an empty answer, which
    # is the difference between "not asked" and "asked, found none".
    with _env(claude=False, codex=True, codex_records=records) as (root, _paths):
        shared_only, nothing = codex_rollups(None, None)
        _, empty = codex_rollups({}, None)
    assert nothing is None
    assert empty is not None and empty["sessions_scanned"] == shared_only["sessions_scanned"]


def test_a_reused_codex_extract_equals_a_fresh_parse():
    """The rollups' cached unit is deliberately *not* their answer: usage is attributed to
    installed skills, so caching the attributed figure would mean installing a skill did
    not change the result. The extract keeps the paths a session read, raw, and the
    inventory is applied when they are merged — so the same cache serves a different
    inventory correctly."""
    from agent_profiler.events import codex_rollups

    skill = "/x/.codex/skills/fourth-skill/SKILL.md"
    other = "/x/.codex/skills/other/SKILL.md"
    records = _codex_records() + [_exec_call(f"cat {skill}", "c1")]
    with _cache_file(), _env(claude=False, codex=True, codex_records=records) as (_r, _p):
        cold = codex_rollups({"user/codex/fourth-skill": skill}, None)
        warm = codex_rollups({"user/codex/fourth-skill": skill}, None)
        # Same cached extract, a different inventory: the read must now attribute to
        # nothing, which a cache of the *answer* could not have got right.
        elsewhere = codex_rollups({"user/codex/other": other}, None)

    assert cold[0]["reuse"]["sessions_parsed"] == 1
    assert warm[0]["reuse"]["sessions_reused"] == 1
    assert cold[1]["reads"] == warm[1]["reads"] == {"user/codex/fourth-skill": 1}
    assert elsewhere[0]["reuse"]["sessions_reused"] == 1
    assert elsewhere[1]["reads"] == {}
    assert elsewhere[1]["sessions_scanned"] == 1


def _probe_records(prompt):
    """A session whose only prompt is one the profiler sends: a probe, not the user's work."""
    base = {"sessionId": "pppppppp-bbbb-cccc-dddd-eeeeeeeeeeee", "version": "2.1.325",
            "cwd": "/proj", "gitBranch": "main"}
    usage = {"input_tokens": 5, "cache_read_input_tokens": 50,
             "cache_creation_input_tokens": 5, "output_tokens": 3}
    msg = {"id": "msg_p", "role": "assistant", "model": "claude-opus-5-5",
           "stop_reason": "end_turn", "usage": usage}
    return [
        dict(base, type="api-request", id="apireq-p", timestamp="2026-11-18T10:00:00.000Z"),
        dict(base, type="user", timestamp="2026-11-18T10:00:00.500Z",
             message={"role": "user", "content": [{"type": "text", "text": prompt}]}),
        dict(base, type="assistant", apiBlockIndex=0, requestRef="apireq-p",
             timestamp="2026-11-18T10:00:05.000Z",
             message=dict(msg, content=[{"type": "text", "text": "X"}])),
    ]


def test_the_normalizer_recognises_a_session_this_tool_wrote_itself():
    """The collector already did; the normalized model did not, so analyze inherited it.

    Measuring a schema means running `claude -p`, which the harness records like any other
    session. Those runs accumulate, and a request count is exactly the denominator an
    always-on prefix cost gets multiplied by — so counting the tool's own exhaust inflates
    every carried figure it reports.
    """
    from agent_profiler.traces import SELF_PROBE_PROMPT
    with tempfile.TemporaryDirectory() as d:
        probe = Path(d) / "probe.jsonl"
        probe.write_text("\n".join(json.dumps(r) for r in _probe_records(SELF_PROBE_PROMPT))
                         + "\n", encoding="utf-8")
        mine = normalize_session(str(probe), "claude_code")
        real = Path(d) / "real.jsonl"
        real.write_text("\n".join(json.dumps(r) for r in _probe_records("fix the parser"))
                        + "\n", encoding="utf-8")
        theirs = normalize_session(str(real), "claude_code")

    assert mine["observed"]["origin"]["self_generated"] is True
    assert mine["observed"]["origin"]["produced_by"] == "agent_profiler_probe"
    assert theirs["observed"]["origin"] is None


def test_a_user_prompt_event_carries_no_user_prompt_text():
    """A privacy guard, because `--events-out` writes this stream to disk.

    Recognising the profiler's own sessions needs the prompt, so the normalizer reads it
    and keeps only two derived signals: a stage name of this tool, or None, and a flag for
    a harness-written interruption marker. Neither is user content. Nothing else about the
    prompt may survive into the event, and a future convenience field is exactly how that
    would quietly stop being true.
    """
    # Deliberately not credential-shaped: a real-looking key in a fixture trips this
    # repo's own pre-commit scanner and everyone else's, and the guard does not need one.
    # What it needs is a distinctive string that must not survive into an event.
    secret = "the spare key is under the third plant pot, do not write this down"
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "s.jsonl"
        p.write_text("\n".join(json.dumps(r) for r in _probe_records(secret)) + "\n",
                     encoding="utf-8")
        out = normalize_session(str(p), "claude_code")

    prompts = [e for e in out["events"] if e["type"] == "user_prompt"]
    assert len(prompts) == 1
    blob = json.dumps(prompts[0], ensure_ascii=False)
    assert "third plant pot" not in blob and "do not write this down" not in blob
    assert prompts[0]["produced_by"] is None
    assert prompts[0]["interruption"] is False
    # The whole stream, not just that one event: no other record may carry it either.
    assert "third plant pot" not in json.dumps(out["observed"], ensure_ascii=False)


def test_the_request_count_excludes_the_profilers_own_probes():
    from agent_profiler.traces import SELF_PROBE_PROMPT
    with _cache_file(), _env(codex=False) as (root, _paths):
        home = Path(os.environ["AGENT_PROFILER_CLAUDE_HOME"])
        key = str(Path(root).resolve()).replace("/", "-").replace(".", "-")
        probe = home / "projects" / key / "probe.jsonl"
        probe.write_text("\n".join(json.dumps(r) for r in _probe_records(SELF_PROBE_PROMPT))
                         + "\n", encoding="utf-8")
        out = normalize_root(str(root), whole_environment=True, use_cache=False)

    obs = out["observed"]
    # The real session answers once; the probe's response is not the user's traffic.
    assert obs["requests"]["by_harness"]["claude_code"] == 1
    assert obs["requests"]["self_generated"]["claude_code"] == 1
    assert obs["self_generated"]["sessions"] == 1
    assert obs["self_generated"]["by"] == {"agent_profiler_probe": 1}
    # Counted, never hidden: the session is still normalized and still in the event totals.
    assert obs["sessions_normalized"] == 2
    assert obs["event_types"]["model_response"] == 2


def test_the_two_walkers_apply_one_rule_for_whose_session_it_is():
    """The collector and the normalizer decide this separately; drift sends the profiler's
    sessions back to the user, so the rule itself lives in one function."""
    from agent_profiler import traces as tr_mod
    assert tr_mod.self_generated_verdict("agent_profiler_probe", "because", 1, 0)
    # A second real prompt is a person typing, whichever walker sees it.
    assert tr_mod.self_generated_verdict("agent_profiler_probe", "because", 1, 1) is None
    # More turns than the collector retains prompts for: declined rather than guessed.
    assert tr_mod.self_generated_verdict("agent_profiler_probe", "because", 9, 0) is None
    assert tr_mod.self_generated_verdict(None, None, 1, 0) is None


def _token_count(total, *, detailed=True, info=True):
    """A Codex `event_msg/token_count` record, in both of its observed shapes."""
    if not info:
        # 0.167 reuses the event type for rate-limit notices: no `info`, `rate_limits`
        # instead. Not a token report.
        return {"timestamp": "2026-11-18T10:00:09.000Z", "type": "event_msg",
                "payload": {"type": "token_count", "info": None,
                            "rate_limits": {"limit_id": "codex", "primary": None}}}
    block = {"input_tokens": total - 100 if detailed else 0,
             "cached_input_tokens": 500 if detailed else 0,
             "cache_write_input_tokens": 0,
             "output_tokens": 100 if detailed else 0,
             "reasoning_output_tokens": 0,
             "total_tokens": total}
    return {"timestamp": "2026-11-18T10:00:09.000Z", "type": "event_msg",
            "payload": {"type": "token_count", "info": {
                "total_token_usage": block, "last_token_usage": block,
                "model_context_window": 258400}}}


def _codex_with(records):
    base = [r for r in _codex_records() if r["type"] == "session_meta"]
    return base + list(records)


def test_the_cumulative_token_record_is_read_as_a_session_total():
    """The guard against double counting had become a blind spot.

    `token_usage_record` exists only in Codex 0.162; `event_msg/token_count` is written by
    every version observed. Refusing to sum the cumulative record was right, but
    discarding it reported *zero tokens* for the large majority of real sessions, hiding
    several times the volume that was being counted.
    """
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "r.jsonl"
        p.write_text("\n".join(json.dumps(r) for r in _codex_with(
            [_token_count(1000), _token_count(2500), _token_count(4000)])) + "\n",
            encoding="utf-8")
        out = normalize_session(str(p), "codex")

    rep = out["observed"]["tokens_reported"]
    # The last one, not the sum: 1000 + 2500 + 4000 would be the same mistake inverted.
    assert rep["total_tokens"] == 4000
    assert rep["cumulative"] is True
    assert rep["source"] == "event_msg/token_count"
    assert rep["breakdown_present"] is True
    # And it must not have leaked into the per-response sums, which are a different
    # quantity measured a different way.
    assert out["observed"]["tokens"]["input_total"] == 0


def test_a_rate_limit_notice_cannot_overwrite_the_session_total():
    """Found by the new verify check failing on the machine that wrote it.

    0.167 emits `token_count` with `info: null` for rate limits. Treating that as an empty
    snapshot lets a notice arriving last replace a real total with zero. No session in the
    real corpus happens to end with one, which is luck, not a design.
    """
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "r.jsonl"
        p.write_text("\n".join(json.dumps(r) for r in _codex_with(
            [_token_count(4000), _token_count(0, info=False)])) + "\n", encoding="utf-8")
        out = normalize_session(str(p), "codex")

    assert out["observed"]["tokens_reported"]["total_tokens"] == 4000
    # The notice is not a cost report at all, so it is not emitted as one.
    assert out["observed"]["event_types"]["cost_report"] == 1


def test_a_version_that_zeroes_the_breakdown_says_so_rather_than_reporting_zeros():
    """Codex 0.166+ zeroes every field but `total_tokens`.

    Reporting those zeros as measurements would read as "nothing was cached", which is a
    claim, not an observation.
    """
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "r.jsonl"
        p.write_text("\n".join(json.dumps(r) for r in _codex_with(
            [_token_count(856161, detailed=False)])) + "\n", encoding="utf-8")
        out = normalize_session(str(p), "codex")

    rep = out["observed"]["tokens_reported"]
    assert rep["breakdown_present"] is False
    assert rep["total_tokens"] == 856161
    assert rep["cached_input_tokens"] == 0


def test_a_session_with_no_token_record_at_all_reports_none_not_zero():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "r.jsonl"
        p.write_text("\n".join(json.dumps(r) for r in _codex_with([])) + "\n",
                     encoding="utf-8")
        out = normalize_session(str(p), "codex")
    assert out["observed"]["tokens_reported"] is None


def test_token_evidence_coverage_is_reported_so_a_total_carries_its_denominator():
    """A figure built from the sessions that happen to carry a record says nothing about
    the ones that do not, and here that was 91% of them."""
    with _cache_file(), _env(claude=False, codex=True) as (root, paths):
        codex_dir = paths["codex"].parent
        rich = codex_dir / "rollout-2026-11-18T10-00-01-cx-2.jsonl"
        recs = _codex_with([_token_count(4000)])
        recs[0]["payload"]["cwd"] = str(Path(root).resolve())
        rich.write_text("\n".join(json.dumps(r) for r in recs) + "\n", encoding="utf-8")
        thin = codex_dir / "rollout-2026-11-18T10-00-02-cx-3.jsonl"
        recs2 = _codex_with([_token_count(99, detailed=False)])
        recs2[0]["payload"]["cwd"] = str(Path(root).resolve())
        thin.write_text("\n".join(json.dumps(r) for r in recs2) + "\n", encoding="utf-8")
        out = normalize_root(str(root), whole_environment=True, use_cache=False)

    cov = out["observed"]["token_coverage"]
    assert cov.get("session_total_with_breakdown") == 1
    assert cov.get("session_total_only") == 1
    rep = out["observed"]["tokens_reported"]
    # Summed across sessions, which is sound; never within one, which would multiply it.
    assert rep["total_tokens"] == 4000 + 99


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"{len(tests)} normalized event regression tests passed")


if __name__ == "__main__":
    main()
