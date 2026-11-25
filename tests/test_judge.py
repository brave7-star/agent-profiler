"""Regression suite for the semantic judge.

No test here spends a token: `_invoke` is replaced with a stub. What is tested is the
funnel, the budget, the redaction, the caching and — above all — that a judge verdict is
never presented as a measurement.
"""

import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path

from agent_profiler import judge as judge_mod
from agent_profiler.cli import main as cli_main
from agent_profiler.judge import BUDGET_TIERS, _redact, judge_root

SESSION = "99999999-8888-7777-6666-555555555555"


def _rec(**kw):
    base = {"sessionId": SESSION, "version": "2.1.325", "cwd": "/proj", "gitBranch": "main"}
    base.update(kw)
    return base


def _assistant(msg_id, blocks, *, ts="2026-11-18T10:00:01.000Z"):
    return _rec(type="assistant", timestamp=ts, message={
        "id": msg_id, "role": "assistant", "model": "claude-opus-5-5", "stop_reason": "tool_use",
        "usage": {"input_tokens": 10, "cache_read_input_tokens": 100,
                  "cache_creation_input_tokens": 0, "output_tokens": 5},
        "content": blocks})


def _call(call_id, tool, **inp):
    return {"type": "tool_use", "id": call_id, "name": tool, "input": inp}


def _result(call_id, *, error=False):
    return _rec(type="user", timestamp="2026-11-18T10:00:02.000Z", message={"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": call_id, "is_error": error, "content": "x"}]})


def _repeat_trace(*, between_commands, target="/proj/shot.png", tool="Read"):
    """A Read, then N Bash commands, then the identical Read again."""
    records = [
        _rec(type="user", timestamp="2026-11-18T10:00:00.000Z",
             message={"role": "user", "content": [{"type": "text", "text": "go"}]}),
        _assistant("m0", [_call("t0", tool, file_path=target)]),
        _result("t0"),
    ]
    for i, command in enumerate(between_commands):
        records.append(_assistant(f"mb{i}", [_call(f"tb{i}", "Bash", command=command)]))
        records.append(_result(f"tb{i}"))
    records.append(_assistant("m9", [_call("t9", tool, file_path=target)]))
    records.append(_result("t9"))
    return records


@contextmanager
def _env(records, *, stub=None):
    with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as h:
        root, home = Path(d), Path(h)
        key = str(root.resolve()).replace("/", "-").replace(".", "-")
        proj = home / ".claude" / "projects" / key
        proj.mkdir(parents=True)
        (proj / f"{SESSION}.jsonl").write_text(
            "\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
        cache = Path(h) / "judge-cache.json"
        saved_env = {k: os.environ.get(k) for k in
                     ("AGENT_PROFILER_HOME", "AGENT_PROFILER_CLAUDE_HOME",
                      "AGENT_PROFILER_CODEX_HOME", "AGENT_PROFILER_JUDGE_CACHE")}
        os.environ["AGENT_PROFILER_HOME"] = str(home)
        os.environ["AGENT_PROFILER_CLAUDE_HOME"] = str(home / ".claude")
        os.environ["AGENT_PROFILER_CODEX_HOME"] = str(home / ".codex")
        os.environ["AGENT_PROFILER_JUDGE_CACHE"] = str(cache)
        original = judge_mod._invoke
        calls = []

        def fake_invoke(prompt, timeout):
            calls.append(prompt)
            return (stub or _default_stub)(prompt)

        judge_mod._invoke = fake_invoke
        try:
            yield root, calls
        finally:
            judge_mod._invoke = original
            for k, v in saved_env.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v


def _ids(prompt):
    import re
    return re.findall(r"id=(\w+)", prompt)


def _default_stub(prompt):
    return {
        "ok": True,
        "verdicts": [{"id": i, "verdict": "unnecessary", "confidence": "high", "why": "commands do not touch target"}
                     for i in _ids(prompt)],
        "model": ["claude-opus-5-5"],
        "cost": {"input_tokens": 500, "cache_read_input_tokens": 0,
                 "cache_creation_input_tokens": 12000, "output_tokens": 80, "cost_usd": 0.05},
        "elapsed_ms": 1200.0,
    }


# -- the funnel --------------------------------------------------------------------------


def test_only_opaque_blocked_repeats_reach_the_judge():
    """A repeat the deterministic layer can already explain must not cost a token."""
    records = _repeat_trace(between_commands=["echo hi"])
    # Add a write to the same target, which is a sufficient explanation on its own.
    records.insert(4, _assistant("mw", [_call("tw", "Edit", file_path="/proj/shot.png")]))
    records.insert(5, _result("tw"))
    with _env(records) as (root, calls):
        r = judge_root(str(root), harness="claude_code", force=True)
        assert r["funnel"]["suspicious_after_filter"] == 0
        assert calls == []
        assert r["profiler_cost"]["cost_usd"] == 0.0


def test_distant_repeats_are_rejected_before_spending():
    """Derived threshold: many intervening commands means separate work, not a duplicate."""
    with _env(_repeat_trace(between_commands=[f"cmd{i}" for i in range(40)])) as (root, calls):
        r = judge_root(str(root), harness="claude_code", force=True)
        assert r["funnel"]["suspicious_after_filter"] == 0
        assert calls == []


def test_targetless_repeats_are_rejected_before_spending():
    """Without a target there is nothing whose freshness the commands could be read against."""
    recs = [
        _rec(type="user", timestamp="2026-11-18T10:00:00.000Z",
             message={"role": "user", "content": [{"type": "text", "text": "go"}]}),
        _assistant("m0", [_call("t0", "mcp__x__update", status="done")]),
        _result("t0"),
        _assistant("mb", [_call("tb", "Bash", command="echo hi")]),
        _result("tb"),
        _assistant("m9", [_call("t9", "mcp__x__update", status="done")]),
        _result("t9"),
    ]
    with _env(recs) as (root, calls):
        r = judge_root(str(root), harness="claude_code", force=True)
        assert r["funnel"]["suspicious_after_filter"] == 0
        assert calls == []


def test_a_genuine_candidate_is_judged_and_becomes_an_estimated_finding():
    with _env(_repeat_trace(between_commands=["grep -rn foo src/"])) as (root, calls):
        r = judge_root(str(root), harness="claude_code", force=True)
        assert r["funnel"]["suspicious_after_filter"] == 1
        assert r["funnel"]["sent_to_judge"] == 1
        assert len(calls) == 1
        finding = r["findings"][0]
        assert finding["kind"] == "repeat_judged_unnecessary"
        # A reading of evidence is never a measurement.
        assert finding["strength"] == "estimated"
        assert finding["judgment"] == "review_only"
        assert finding["assumptions"]
        assert finding["evidence"]["commands_shown"]
        assert finding["judge"]["confidence"] == "high"


def test_everything_is_batched_into_one_call():
    """Per-candidate calls would multiply a 10k-34k token fixed overhead."""
    records = [_rec(type="user", timestamp="2026-11-18T10:00:00.000Z",
                    message={"role": "user", "content": [{"type": "text", "text": "go"}]})]
    for i in range(5):
        records += [
            _assistant(f"a{i}", [_call(f"r{i}a", "Read", file_path=f"/proj/f{i}.png")]),
            _result(f"r{i}a"),
            _assistant(f"b{i}", [_call(f"c{i}", "Bash", command=f"grep x{i} src/")]),
            _result(f"c{i}"),
            _assistant(f"d{i}", [_call(f"r{i}b", "Read", file_path=f"/proj/f{i}.png")]),
            _result(f"r{i}b"),
        ]
    with _env(records) as (root, calls):
        r = judge_root(str(root), harness="claude_code", force=True)
        assert r["funnel"]["after_dedup"] == 5
        assert len(calls) == 1, "all candidates must share one invocation"
        assert r["profiler_cost"]["llm_calls"] == 1
        assert len(r["findings"]) == 5


def test_identical_evidence_is_deduped_before_judging():
    records = [_rec(type="user", timestamp="2026-11-18T10:00:00.000Z",
                    message={"role": "user", "content": [{"type": "text", "text": "go"}]})]
    # Same tool, same target, same intervening command, twice over.
    for i in range(2):
        records += [
            _assistant(f"a{i}", [_call(f"r{i}a", "Read", file_path="/proj/same.png")]),
            _result(f"r{i}a"),
            _assistant(f"b{i}", [_call(f"c{i}", "Bash", command="echo identical")]),
            _result(f"c{i}"),
        ]
    records += [_assistant("z", [_call("rz", "Read", file_path="/proj/same.png")]), _result("rz")]
    with _env(records) as (root, calls):
        r = judge_root(str(root), harness="claude_code", force=True)
        assert r["funnel"]["suspicious_after_filter"] > r["funnel"]["after_dedup"]
        assert len(_ids(calls[0])) == r["funnel"]["after_dedup"]


# -- budget and cost ---------------------------------------------------------------------


def test_quick_tier_spends_nothing():
    with _env(_repeat_trace(between_commands=["grep x"])) as (root, calls):
        r = judge_root(str(root), harness="claude_code", tier="quick")
        assert calls == []
        assert r["funnel"]["sent_to_judge"] == 0
        assert r["profiler_cost"]["cost_usd"] == 0.0
        assert r["findings"] == []


def test_dry_run_builds_the_funnel_without_spending():
    with _env(_repeat_trace(between_commands=["grep x"])) as (root, calls):
        r = judge_root(str(root), harness="claude_code", dry_run=True)
        assert calls == []
        assert r["funnel"]["suspicious_after_filter"] == 1
        assert r["funnel"]["sent_to_judge"] == 0
        assert r["candidates"] and r["candidates"][0]["between_commands"]
        assert r["profiler_cost"]["llm_input_tokens"] == 0


def test_budget_caps_candidates_before_the_call():
    records = [_rec(type="user", timestamp="2026-11-18T10:00:00.000Z",
                    message={"role": "user", "content": [{"type": "text", "text": "go"}]})]
    for i in range(70):
        records += [
            _assistant(f"a{i}", [_call(f"r{i}a", "Read", file_path=f"/proj/f{i}.png")]),
            _result(f"r{i}a"),
            _assistant(f"b{i}", [_call(f"c{i}", "Bash", command=f"grep x{i}")]),
            _result(f"c{i}"),
            _assistant(f"d{i}", [_call(f"r{i}b", "Read", file_path=f"/proj/f{i}.png")]),
            _result(f"r{i}b"),
        ]
    with _env(records) as (root, calls):
        r = judge_root(str(root), harness="claude_code", tier="standard", force=True)
        cap = BUDGET_TIERS["standard"]["max_candidates"]
        assert r["funnel"]["sent_to_judge"] == cap
        assert r["funnel"]["dropped_over_budget"] == 70 - cap
        assert len(_ids(calls[0])) == cap


def test_self_cost_is_reported_with_a_per_candidate_figure():
    with _env(_repeat_trace(between_commands=["grep x"])) as (root, _calls):
        cost = judge_root(str(root), harness="claude_code", force=True)["profiler_cost"]
        assert cost["llm_calls"] == 1
        assert cost["llm_cache_creation_tokens"] == 12000
        assert cost["cost_usd"] == 0.05
        assert cost["candidates_judged"] == 1
        assert cost["cost_usd_per_candidate"] == 0.05
        assert "invocation_overhead" in cost


def test_verdicts_are_cached_so_a_rerun_is_free():
    with _env(_repeat_trace(between_commands=["grep x"])) as (root, calls):
        first = judge_root(str(root), harness="claude_code", force=True)
        second = judge_root(str(root), harness="claude_code", force=True)
        assert len(calls) == 1, "the second run must not call the judge again"
        assert second["funnel"]["served_from_cache"] == 1
        assert second["funnel"]["sent_to_judge"] == 0
        assert second["profiler_cost"]["cost_usd"] == 0.0
        assert len(second["findings"]) == len(first["findings"]) == 1
        assert second["findings"][0]["judge"]["served_from"] == "cache"


# -- safety and honesty ------------------------------------------------------------------


def test_judge_sees_every_intervening_command():
    """Real FP: 2 verdicts said 'nothing regenerated it' on partially shown evidence."""
    commands = [f"grep x{i} src/" for i in range(9)]
    with _env(_repeat_trace(between_commands=commands)) as (root, calls):
        r = judge_root(str(root), harness="claude_code", force=True)
        finding = r["findings"][0]
        assert finding["evidence_complete"] is True
        assert len(finding["evidence"]["commands_shown"]) == 9
        assert r["observed"]["findings_on_partial_evidence"] == 0
        # Every command must actually reach the prompt.
        for command in commands:
            assert command in calls[0]


def test_partial_evidence_is_declared_in_the_finding():
    """If a command cannot be recovered, the verdict must say the evidence was incomplete."""
    import agent_profiler.judge as jm
    original = jm._raw_command
    jm._raw_command = lambda source, line: None
    try:
        with _env(_repeat_trace(between_commands=["grep a", "grep b"])) as (root, _calls):
            r = judge_root(str(root), harness="claude_code", force=True)
            finding = r["findings"][0]
            assert finding["evidence_complete"] is False
            assert r["observed"]["findings_on_partial_evidence"] == 1
            assert any("Evidence was incomplete" in a for a in finding["assumptions"])
    finally:
        jm._raw_command = original


def test_credentials_are_redacted_before_anything_is_sent():
    secret = "tok_7xKQv2NbRtY8pLmW4sZcH6jD"
    command = f'curl -H "Authorization: Bearer {secret}" https://api.invalid/x'
    with _env(_repeat_trace(between_commands=[command])) as (root, calls):
        r = judge_root(str(root), harness="claude_code", force=True)
        assert secret not in calls[0]
        assert "[REDACTED]" in calls[0]
        assert secret not in json.dumps(r)
        assert r["findings"][0]["evidence"]["commands_shown"][0].count("[REDACTED]") == 1


def test_redaction_patterns():
    for text in ('export API_KEY=abc123xyz', 'mysql --password=hunter2',
                 'Authorization: Bearer sk-ant-aaaaaaaa'):
        cleaned, hit = _redact(text)
        assert hit and "[REDACTED]" in cleaned
    cleaned, hit = _redact("ls -la /tmp")
    assert not hit and cleaned == "ls -la /tmp"
    # Redacted on shape alone, with no `=` and no known vendor prefix to match on, so
    # rotating a key to a different prefix cannot quietly drop it out of coverage.
    token = "tok_7xKQv2NbRtY8pLmW4sZcH6jD"
    cleaned, hit = _redact(f"curl --api-key {token} https://api.invalid/x")
    assert hit and token not in cleaned
    # A long snake_case identifier is not a credential. Redacting it would degrade the
    # evidence the judge reads, so the entropy test has to tell the two apart.
    for benign in ("node_modules/.bin/eslint", "tmp_file_2026_11_24_backup_copy",
                   "my_table_name_in_the_database"):
        cleaned, hit = _redact(benign)
        assert not hit and cleaned == benign


def test_unknown_and_necessary_verdicts_produce_no_finding():
    """'Unknown' is an accepted answer and must not be converted into waste."""
    def stub(prompt):
        out = _default_stub(prompt)
        out["verdicts"] = [dict(v, verdict="unknown", confidence="low") for v in out["verdicts"]]
        return out

    with _env(_repeat_trace(between_commands=["grep x"]), stub=stub) as (root, _calls):
        r = judge_root(str(root), harness="claude_code", force=True)
        assert r["findings"] == []
        assert r["observed"]["verdicts"] == {"unknown": 1}


def test_no_counterfactual_claims_in_output():
    with _env(_repeat_trace(between_commands=["grep x"])) as (root, _calls):
        blob = json.dumps(judge_root(str(root), harness="claude_code", force=True)).lower()
        for banned in ("would save", "could save", "saves ", "useless"):
            assert banned not in blob, banned


def test_judge_failure_degrades_without_findings():
    with _env(_repeat_trace(between_commands=["grep x"]),
              stub=lambda p: {"ok": False, "error": "timeout"}) as (root, _calls):
        r = judge_root(str(root), harness="claude_code", force=True)
        assert r["findings"] == []
        assert r["calls"][0]["ok"] is False
        assert r["observed"]["verdicts"] == {"not_judged": 1}


def test_unparsable_judge_reply_is_not_invented_into_findings():
    with _env(_repeat_trace(between_commands=["grep x"]),
              stub=lambda p: {"ok": True, "verdicts": [], "unparsed": True,
                              "cost": {"cost_usd": 0.02, "input_tokens": 100,
                                       "cache_read_input_tokens": 0,
                                       "cache_creation_input_tokens": 0, "output_tokens": 5}}) as (root, _c):
        r = judge_root(str(root), harness="claude_code", force=True)
        assert r["findings"] == []
        # The spend still gets reported, because it happened.
        assert r["profiler_cost"]["cost_usd"] == 0.02


def test_unknown_tier_is_rejected():
    with _env([]) as (root, _calls):
        try:
            judge_root(str(root), tier="unlimited")
        except ValueError as exc:
            assert "unknown budget tier" in str(exc)
        else:
            raise AssertionError("an unknown tier must be rejected")


def test_cli_dry_run_and_errors():
    with _env(_repeat_trace(between_commands=["grep x"])) as (root, _calls):
        assert cli_main(["judge", "--root", str(root), "--dry-run", "--output", os.devnull]) == 0
        assert cli_main(["judge", "--root", ""]) == 2
        assert cli_main(["judge", "--root", str(root / "nope")]) == 2



# -- the economic gate -------------------------------------------------------------------


def test_the_gate_refuses_a_trade_worth_less_than_asking():
    """Stage 5's own failure, now prevented: one candidate is worth far less than the call
    that would evaluate it, so the call must not be made."""
    with _env(_repeat_trace(between_commands=["grep x"])) as (root, calls):
        r = judge_root(str(root), harness="claude_code")
        assert r["gate"]["decision"] == "refused"
        assert calls == [], "a refused gate must not spend"
        assert r["funnel"]["sent_to_judge"] == 0
        assert r["funnel"]["dropped_by_gate"] == 1
        assert r["profiler_cost"]["cost_usd"] == 0.0
        # A refusal is a result: the funnel still reports what it found.
        assert r["funnel"]["suspicious_after_filter"] == 1


def test_the_gate_approves_a_corpus_where_the_overhead_amortises():
    """The fixed overhead is per-call, so enough candidates make the same call worth it.
    The break-even point is derived from the gate, not chosen."""
    from agent_profiler.judge import MIN_CANDIDATES_WORTH_ASKING
    records = [_rec(type="user", timestamp="2026-11-18T10:00:00.000Z",
                    message={"role": "user", "content": [{"type": "text", "text": "go"}]})]
    for i in range(MIN_CANDIDATES_WORTH_ASKING):
        records += [
            _assistant(f"a{i}", [_call(f"r{i}a", "Read", file_path=f"/proj/f{i}.png")]),
            _result(f"r{i}a"),
            _assistant(f"b{i}", [_call(f"c{i}", "Bash", command=f"grep x{i} src/")]),
            _result(f"c{i}"),
            _assistant(f"d{i}", [_call(f"r{i}b", "Read", file_path=f"/proj/f{i}.png")]),
            _result(f"r{i}b"),
        ]
    with _env(records) as (root, calls):
        r = judge_root(str(root), harness="claude_code")
        assert r["gate"]["decision"] == "approved"
        assert r["gate"]["eig_ratio"] >= r["gate"]["threshold"]
        assert len(calls) == 1
        assert r["funnel"]["dropped_by_gate"] == 0


def test_force_overrides_the_gate_and_says_so():
    with _env(_repeat_trace(between_commands=["grep x"])) as (root, calls):
        r = judge_root(str(root), harness="claude_code", force=True)
        assert r["gate"]["decision"] == "overridden"
        assert "--force was passed" in r["gate"]["reason"]
        assert len(calls) == 1


def test_the_tier_cap_is_derived_from_the_gate_not_chosen():
    """A hand-picked cap of 60 was refused at 4.83x — exactly the kind of number guessing
    produces. The standard tier now starts at the break-even point."""
    from agent_profiler.judge import (BUDGET_TIERS, MIN_CANDIDATES_WORTH_ASKING,
                                      _judge_gate)
    assert _judge_gate(MIN_CANDIDATES_WORTH_ASKING, 0)["decision"] == "approved"
    assert _judge_gate(MIN_CANDIDATES_WORTH_ASKING - 1, 0)["decision"] == "refused"
    assert BUDGET_TIERS["standard"]["max_candidates"] >= MIN_CANDIDATES_WORTH_ASKING


def test_the_gate_values_candidates_from_measured_figures():
    """Both sides of the ratio must be traceable to something this project measured."""
    from agent_profiler.judge import INVOCATION_COST_USD, TOKENS_PER_CANDIDATE, _judge_gate
    gate = _judge_gate(60, 0)
    assert str(TOKENS_PER_CANDIDATE) in gate["basis"]
    assert str(INVOCATION_COST_USD) in gate["basis"]
    low, high = gate["value_usd_range"]
    assert low < high, "cache-read and fresh-input bounds must be stated as a range"


def test_an_all_cached_run_is_refused_without_an_error():
    """Nothing pending is a clean no-op, not a failed trade."""
    with _env(_repeat_trace(between_commands=["grep x"])) as (root, calls):
        judge_root(str(root), harness="claude_code", force=True)
        again = judge_root(str(root), harness="claude_code")
        assert again["gate"]["decision"] == "refused"
        assert "cache already answers" in again["gate"]["reason"]
        assert again["funnel"]["served_from_cache"] == 1
        assert len(again["findings"]) == 1, "cached verdicts are still served"


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"{len(tests)} judge regression tests passed (0 tokens spent)")


if __name__ == "__main__":
    main()
