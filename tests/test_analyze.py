"""Regression suite for the deterministic analyzer.

The suite is weighted towards *not* producing findings. The dogfood that drove this module
found that the canonical "repeated reads are waste" finding never once held on real sessions
once confounds were checked, so most of these tests assert that the analyzer stays quiet
when the evidence does not support a claim.
"""

import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path

from agent_profiler.analyze import analyze_root, analyze_session
from agent_profiler.cli import main as cli_main

SESSION = "dddddddd-eeee-ffff-0000-111111111111"
BILLING_START_MS = 1794996000000


def _rec(**kw):
    base = {"sessionId": SESSION, "version": "2.1.325", "cwd": "/proj", "gitBranch": "main"}
    base.update(kw)
    return base


def _assistant(msg_id, blocks, *, ts, usage=None, stop="tool_use"):
    message = {"id": msg_id, "role": "assistant", "model": "claude-opus-5-5",
               "stop_reason": stop, "content": blocks}
    if usage is not None:
        message["usage"] = usage
    return _rec(type="assistant", message=message, timestamp=ts)


def _usage(inp=10, read=9000, create=100, out=50):
    return {"input_tokens": inp, "cache_read_input_tokens": read,
            "cache_creation_input_tokens": create, "output_tokens": out}


def _call(call_id, tool, **inp):
    return {"type": "tool_use", "id": call_id, "name": tool, "input": inp}


def _result(call_id, *, error=False, ts="2026-11-18T10:00:30.000Z"):
    return _rec(type="user", timestamp=ts, message={"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": call_id, "is_error": error, "content": "x"}]})


def _prompt(ts="2026-11-18T10:00:00.000Z"):
    return _rec(type="user", timestamp=ts,
                message={"role": "user", "content": [{"type": "text", "text": "go"}]})


@contextmanager
def _session(records, *, mcp=None, codex_records=None):
    with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as h:
        root, home = Path(d), Path(h)
        if mcp is not None:
            (root / ".mcp.json").write_text(json.dumps({"mcpServers": mcp}), encoding="utf-8")
        key = str(root.resolve()).replace("/", "-").replace(".", "-")
        proj = home / ".claude" / "projects" / key
        proj.mkdir(parents=True)
        path = proj / f"{SESSION}.jsonl"
        path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
        if codex_records is not None:
            xdir = home / ".codex" / "sessions" / "2026" / "11" / "18"
            xdir.mkdir(parents=True)
            for i, rec in enumerate(codex_records):
                if rec.get("type") == "session_meta":
                    codex_records[i]["payload"]["cwd"] = str(root.resolve())
            (xdir / "rollout-2026-11-18T10-00-00-cx.jsonl").write_text(
                "\n".join(json.dumps(r) for r in codex_records) + "\n", encoding="utf-8")
        saved = {k: os.environ.get(k) for k in
                 ("AGENT_PROFILER_HOME", "AGENT_PROFILER_CLAUDE_HOME", "AGENT_PROFILER_CODEX_HOME")}
        os.environ["AGENT_PROFILER_HOME"] = str(home)
        os.environ["AGENT_PROFILER_CLAUDE_HOME"] = str(home / ".claude")
        os.environ["AGENT_PROFILER_CODEX_HOME"] = str(home / ".codex")
        try:
            yield root, path
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v


def _kinds(result):
    return {f["kind"] for f in result["findings"]}


def _find(result, kind):
    return next(f for f in result["findings"] if f["kind"] == kind)


# -- repetition: the heart of the stage -------------------------------------------------


def test_clean_repeat_is_reported_as_observed():
    """Nothing between two identical Reads, so the repeat is genuinely avoidable."""
    records = [
        _prompt(),
        _assistant("m1", [_call("t1", "Read", file_path="/proj/a.py")],
                   ts="2026-11-18T10:00:01.000Z", usage=_usage()),
        _result("t1"),
        _assistant("m2", [_call("t2", "Read", file_path="/proj/a.py")],
                   ts="2026-11-18T10:00:02.000Z", usage=_usage()),
        _result("t2"),
    ]
    with _session(records) as (_root, path):
        r = analyze_session(str(path))
        assert "repeated_identical_call" in _kinds(r)
        f = _find(r, "repeated_identical_call")
        assert f["strength"] == "observed"
        assert f["evidence"]["repeats"] == 1
        assert f["evidence"]["detail"][0]["tool"] == "Read"
        assert "opaque_command_between" in f["evidence"]["ruled_out"]


def test_opaque_command_between_makes_a_repeat_inconclusive():
    """The real-corpus case: a Bash call may have regenerated the file being re-read."""
    records = [
        _prompt(),
        _assistant("m1", [_call("t1", "Read", file_path="/proj/shot.png")],
                   ts="2026-11-18T10:00:01.000Z", usage=_usage()),
        _result("t1"),
        _assistant("m2", [_call("t2", "Bash", command="make screenshot")],
                   ts="2026-11-18T10:00:02.000Z", usage=_usage()),
        _result("t2"),
        _assistant("m3", [_call("t3", "Read", file_path="/proj/shot.png")],
                   ts="2026-11-18T10:00:03.000Z", usage=_usage()),
        _result("t3"),
    ]
    with _session(records) as (_root, path):
        r = analyze_session(str(path))
        assert "repeated_identical_call" not in _kinds(r)
        f = _find(r, "repeated_call_inconclusive")
        assert f["judgment"] == "not_waste"
        assert f["evidence"]["reasons"] == {"opaque_command_between": 1}


def test_write_to_the_same_target_makes_a_repeat_inconclusive():
    records = [
        _prompt(),
        _assistant("m1", [_call("t1", "Read", file_path="/proj/a.py")],
                   ts="2026-11-18T10:00:01.000Z", usage=_usage()),
        _result("t1"),
        _assistant("m2", [_call("t2", "Edit", file_path="/proj/a.py")],
                   ts="2026-11-18T10:00:02.000Z", usage=_usage()),
        _result("t2"),
        _assistant("m3", [_call("t3", "Read", file_path="/proj/a.py")],
                   ts="2026-11-18T10:00:03.000Z", usage=_usage()),
        _result("t3"),
    ]
    with _session(records) as (_root, path):
        r = analyze_session(str(path))
        assert "repeated_identical_call" not in _kinds(r)
        assert _find(r, "repeated_call_inconclusive")["evidence"]["reasons"] == {"target_written_between": 1}


def test_edit_to_a_different_file_does_not_excuse_the_repeat():
    """The confound must be specific: an unrelated write is not an explanation."""
    records = [
        _prompt(),
        _assistant("m1", [_call("t1", "Read", file_path="/proj/a.py")],
                   ts="2026-11-18T10:00:01.000Z", usage=_usage()),
        _result("t1"),
        _assistant("m2", [_call("t2", "Edit", file_path="/proj/OTHER.py")],
                   ts="2026-11-18T10:00:02.000Z", usage=_usage()),
        _result("t2"),
        _assistant("m3", [_call("t3", "Read", file_path="/proj/a.py")],
                   ts="2026-11-18T10:00:03.000Z", usage=_usage()),
        _result("t3"),
    ]
    with _session(records) as (_root, path):
        r = analyze_session(str(path))
        assert "repeated_identical_call" in _kinds(r)


def test_compaction_between_makes_a_repeat_inconclusive():
    records = [
        _prompt(),
        _assistant("m1", [_call("t1", "Read", file_path="/proj/a.py")],
                   ts="2026-11-18T10:00:01.000Z", usage=_usage()),
        _result("t1"),
        _rec(type="compact-boundary", timestamp="2026-11-18T10:00:02.000Z"),
        _assistant("m2", [_call("t2", "Read", file_path="/proj/a.py")],
                   ts="2026-11-18T10:00:03.000Z", usage=_usage()),
        _result("t2"),
    ]
    with _session(records) as (_root, path):
        r = analyze_session(str(path))
        assert "repeated_identical_call" not in _kinds(r)
        assert _find(r, "repeated_call_inconclusive")["evidence"]["reasons"] == {"context_compacted_between": 1}


def test_retry_after_failure_is_its_own_category():
    """Repeating a call that failed is a retry, not avoidable duplication."""
    records = [
        _prompt(),
        _assistant("m1", [_call("t1", "Read", file_path="/proj/a.py")],
                   ts="2026-11-18T10:00:01.000Z", usage=_usage()),
        _result("t1", error=True),
        _assistant("m2", [_call("t2", "Read", file_path="/proj/a.py")],
                   ts="2026-11-18T10:00:02.000Z", usage=_usage()),
        _result("t2"),
    ]
    with _session(records) as (_root, path):
        r = analyze_session(str(path))
        kinds = _kinds(r)
        assert "retry_after_failure" in kinds
        assert "repeated_identical_call" not in kinds
        assert r["observed"]["repeats"]["verdicts"] == {"retry_after_failure": 1}


def test_different_inputs_are_not_a_repeat():
    records = [
        _prompt(),
        _assistant("m1", [_call("t1", "Read", file_path="/proj/a.py")],
                   ts="2026-11-18T10:00:01.000Z", usage=_usage()),
        _result("t1"),
        _assistant("m2", [_call("t2", "Read", file_path="/proj/b.py")],
                   ts="2026-11-18T10:00:02.000Z", usage=_usage()),
        _result("t2"),
    ]
    with _session(records) as (_root, path):
        r = analyze_session(str(path))
        assert not ({"repeated_identical_call", "repeated_call_inconclusive"} & _kinds(r))
        assert r["observed"]["repeats"]["verdicts"] == {}


# -- restraint ---------------------------------------------------------------------------


def test_small_error_counts_do_not_produce_a_rate_finding():
    """A failure rate over a handful of calls is noise, not a finding."""
    records = [_prompt(),
               _assistant("m1", [_call("t1", "Bash", command="false")],
                          ts="2026-11-18T10:00:01.000Z", usage=_usage()),
               _result("t1", error=True)]
    with _session(records) as (_root, path):
        assert "tool_failures" not in _kinds(analyze_session(str(path)))


def test_error_rate_reported_once_there_are_enough_calls():
    records = [_prompt()]
    for i in range(25):
        records.append(_assistant(f"m{i}", [_call(f"t{i}", "Bash", command=f"cmd{i}")],
                                  ts="2026-11-18T10:00:01.000Z", usage=_usage()))
        records.append(_result(f"t{i}", error=(i < 3)))
    with _session(records) as (_root, path):
        f = _find(analyze_session(str(path)), "tool_failures")
        assert f["strength"] == "observed"
        assert f["evidence"]["errors"] == 3 and f["evidence"]["calls"] == 25
        assert f["evidence"]["rate_pct"] == 12.0
        assert f["references"] and f["references"][0]["tool"] == "Bash"


def test_low_cache_hit_is_estimated_and_carries_assumptions():
    """Cache hit tracks session length, so the finding must not imply waste."""
    records = [_prompt()]
    for i in range(10):
        records.append(_assistant(f"m{i}", [{"type": "text", "text": "."}],
                                  ts="2026-11-18T10:00:01.000Z",
                                  usage=_usage(inp=900, read=100, create=0, out=10), stop="end_turn"))
    with _session(records) as (_root, path):
        f = _find(analyze_session(str(path)), "low_cache_hit_rate")
        assert f["strength"] == "estimated"
        assert f["evidence"]["cache_hit_pct"] == 10.0
        assert len(f["assumptions"]) >= 2
        assert any("session length" in a for a in f["assumptions"])


def test_short_session_gets_no_cache_finding():
    records = [_prompt(),
               _assistant("m1", [{"type": "text", "text": "."}], ts="2026-11-18T10:00:01.000Z",
                          usage=_usage(inp=900, read=0, create=0, out=10), stop="end_turn")]
    with _session(records) as (_root, path):
        assert "low_cache_hit_rate" not in _kinds(analyze_session(str(path)))


def test_no_findings_claim_a_counterfactual():
    """Nothing in this stage may say removing something would save anything."""
    records = [_prompt()]
    for i in range(25):
        records.append(_assistant(f"m{i}", [_call(f"t{i}", "Bash", command=f"c{i}")],
                                  ts="2026-11-18T10:00:01.000Z",
                                  usage=_usage(inp=900, read=10, create=0, out=5)))
        records.append(_result(f"t{i}", error=(i % 5 == 0)))
    with _session(records, mcp={"unused-server": {"type": "http"}}) as (root, _path):
        r = analyze_root(str(root))
        blob = json.dumps(r).lower()
        for banned in ("would save", "could save", "saves ", "wasted ", "useless"):
            assert banned not in blob, banned
        assert set(f["strength"] for f in r["findings"]) <= {"observed", "estimated"}
        assert all(f["judgment"] in {"review_only", "not_waste"} for f in r["findings"])


# -- Installed -> Used bridge -----------------------------------------------------------


def test_unused_mcp_requires_the_declaring_harness_to_be_traced():
    """The dogfood FP: a Codex-only server is not 'unused' because Claude never called it."""
    records = [_prompt(),
               _assistant("m1", [_call("t1", "Read", file_path="/proj/a.py")],
                          ts="2026-11-18T10:00:01.000Z", usage=_usage()),
               _result("t1")]
    with _session(records, mcp={"claude-side": {"type": "http"}}) as (root, _path):
        # Only Claude Code traces exist, and .mcp.json is a Claude declaration.
        r = analyze_root(str(root), harness="claude_code")
        f = _find(r, "declared_mcp_never_called")
        assert [s["name"] for s in f["evidence"]["servers"]] == ["claude-side"]
        assert f["evidence"]["harnesses_analyzed"] == ["claude_code"]
        assert "not evidence of low value" in f["limits"]


def test_called_servers_produce_no_finding():
    records = [_prompt(),
               _assistant("m1", [_call("t1", "mcp__used-server__do_thing", x=1)],
                          ts="2026-11-18T10:00:01.000Z", usage=_usage()),
               _result("t1")]
    with _session(records, mcp={"used-server": {"type": "http"}}) as (root, _path):
        r = analyze_root(str(root), harness="claude_code")
        assert "declared_mcp_never_called" not in _kinds(r)


def test_codex_only_server_is_reported_as_not_assessable():
    """Real finding: 3 user-level Codex servers looked 'unused' from Claude traces."""
    records = [_prompt(),
               _assistant("m1", [_call("t1", "Read", file_path="/proj/a.py")],
                          ts="2026-11-18T10:00:01.000Z", usage=_usage()),
               _result("t1")]
    with _session(records) as (root, _path):
        (root / ".codex").mkdir()
        (root / ".codex" / "config.toml").write_text(
            "model = 'gpt'\n[mcp_servers.codex-only]\ncommand = 'x'\n", encoding="utf-8")
        r = analyze_root(str(root), harness="claude_code")
        f = _find(r, "declared_mcp_not_traced")
        assert [s["name"] for s in f["evidence"]["servers"]] == ["codex-only"]
        assert f["judgment"] == "not_waste"
        assert "declared_mcp_never_called" not in _kinds(r)


# -- contract ---------------------------------------------------------------------------


def test_analyzer_consumes_events_not_transcripts():
    """Guard the stage boundary: the analyzer must not parse raw records itself."""
    source = Path("src/agent_profiler/analyze.py").read_text(encoding="utf-8")
    for raw_marker in ("api-request-blob", "requestRef", "cost-state", "token_usage_record",
                       "apiBlockIndex", "cache_read_input_tokens", "custom_tool_call"):
        assert raw_marker not in source, raw_marker
    assert "json.loads" not in source


def test_findings_carry_strength_and_evidence():
    records = [_prompt()]
    for i in range(25):
        records.append(_assistant(f"m{i}", [_call(f"t{i}", "Bash", command=f"c{i}")],
                                  ts="2026-11-18T10:00:01.000Z", usage=_usage()))
        records.append(_result(f"t{i}", error=(i < 4)))
    with _session(records) as (_root, path):
        r = analyze_session(str(path))
        assert r["findings"]
        for f in r["findings"]:
            assert f["strength"] in {"observed", "estimated"}
            assert f["summary"] and isinstance(f["evidence"], dict)
            assert f["scope"] in {"session", "run", "root"}
            if f["strength"] == "estimated":
                assert f["assumptions"]


def test_profiler_cost_is_zero_tokens():
    with _session([_prompt()]) as (root, path):
        for result in (analyze_session(str(path)), analyze_root(str(root))):
            cost = result["profiler_cost"]
            assert cost["llm_input_tokens"] == 0 and cost["llm_output_tokens"] == 0


def test_empty_session_yields_no_findings():
    with _session([]) as (_root, path):
        r = analyze_session(str(path))
        assert r["findings"] == []
        assert r["observed"]["tools"]["calls_total"] == 0


def test_cli_exit_codes():
    with _session([_prompt()]) as (root, path):
        assert cli_main(["analyze", "--session", str(path), "--output", os.devnull]) == 0
        assert cli_main(["analyze", "--root", str(root), "--output", os.devnull]) == 0
        assert cli_main(["analyze", "--root", str(root), "--full", "--output", os.devnull]) == 0
        assert cli_main(["analyze", "--root", ""]) == 2
        assert cli_main(["analyze", "--session", str(root / "nope.jsonl")]) == 2


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"{len(tests)} analyzer regression tests passed")


if __name__ == "__main__":
    main()
